"""通过 Git Credential Manager 已登录的账户发布 GitHub Pages；不保存或打印凭据。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OWNER = 'bdlfy114514'
REPOSITORY = 'naifamily-classifier'
REMOTE = f'https://github.com/{OWNER}/{REPOSITORY}.git'
TRACKED = ['.gitignore','README.md','TRAINING_QUICKSTART.md','LICENSE','reports','package.json','common.py','model.py','preprocess.py',
           'preprocess_config.json','train.py','test.py','export_web.py','setup_env.ps1',
           'requirements.txt','requirements-lock.txt','requirements-web.txt','scripts','tests','docs','show']


def git(*args, check=True):
    env = {**os.environ, 'GIT_TERMINAL_PROMPT':'0', 'GCM_INTERACTIVE':'never'}
    result = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True,
                            encoding='utf8', errors='replace', env=env)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or 'Git operation failed')
    return result


def credentials():
    result = subprocess.run(['git','-c','credential.interactive=never','credential','fill'],
        input=f'protocol=https\nhost=github.com\nusername={OWNER}\n\n', capture_output=True,
        text=True, cwd=ROOT, timeout=30,
        env={**os.environ,'GIT_TERMINAL_PROMPT':'0','GCM_INTERACTIVE':'never'})
    if result.returncode:
        return None
    fields = dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    return fields.get('password')


def api(token, path, method='GET', data=None, allowed=(200,201,202,204)):
    request = urllib.request.Request('https://api.github.com'+path, method=method,
        data=json.dumps(data).encode() if data is not None else None,
        headers={'Authorization':'Bearer '+token, 'Accept':'application/vnd.github+json',
                 'X-GitHub-Api-Version':'2022-11-28','User-Agent':'naifamily-classifier-publisher',
                 'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=30) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as error:
        payload = json.loads(error.read() or b'{}')
        if error.code in allowed:
            return error.code, payload
        raise RuntimeError(f'GitHub API {error.code}: {payload.get("message","Request failed")}') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status',action='store_true',help='只查看登录/仓库/Pages 状态')
    args = parser.parse_args()
    token = credentials()
    if not token:
        print(json.dumps({'authenticated':False,'next':'git credential-manager github login --device --no-ui --username bdlfy114514'}))
        raise SystemExit(2)
    _, account = api(token,'/user')
    if account['login'].lower() != OWNER.lower():
        raise RuntimeError(f'登录账户是 {account["login"]}，目标账户是 {OWNER}；未执行任何发布操作。')
    repo_path = f'/repos/{OWNER}/{REPOSITORY}'
    status, repo = api(token,repo_path,allowed=(200,404))
    if args.status:
        if status == 404:
            print(json.dumps({'authenticated':True,'login':account['login'],'repositoryExists':False}))
            return
        pages_status, pages = api(token,repo_path+'/pages',allowed=(200,404))
        output = {'authenticated':True,'login':account['login'],'repository':repo['html_url'],
                  'pagesConfigured':pages_status==200,'pagesUrl':pages.get('html_url'),'pagesStatus':pages.get('status')}
        if pages_status == 200:
            build_status, build = api(token,repo_path+'/pages/builds/latest',allowed=(200,404))
            output['latestBuild'] = build.get('status') if build_status==200 else None
            output['buildError'] = build.get('error',{}).get('message')
        print(json.dumps(output))
        return
    subprocess.run(['node','--test','tests/web.test.mjs'],cwd=ROOT,check=True)
    if status == 404:
        _, repo = api(token,'/user/repos','POST',{'name':REPOSITORY,'description':'上传图片，识别 11 种奶家人。图片仅在浏览器本地处理。',
            'private':False,'auto_init':False})
    if repo['owner']['login'].lower() != OWNER.lower() or repo['private']:
        raise RuntimeError('仓库所有者或公开性与 Pages 发布配置不符；停止发布。')
    origin = git('remote','get-url','origin',check=False)
    if origin.returncode:
        git('remote','add','origin',REMOTE)
    elif origin.stdout.strip().removesuffix('.git') != REMOTE.removesuffix('.git'):
        raise RuntimeError('现有 origin 指向不同仓库，未修改远程配置。')
    git('branch','-M','main')
    git('add','--',*TRACKED)
    changes = git('diff','--cached','--quiet',check=False)
    if changes.returncode == 1:
        git('-c',f'user.name={OWNER}','-c',f'user.email={account["id"]}+{OWNER}@users.noreply.github.com',
            'commit','-m','Build browser-based Nai Family classifier')
    git('push','-u','origin','main')
    page_status, page = api(token,repo_path+'/pages',allowed=(200,404))
    source = {'branch':'main','path':'/docs'}
    if page_status == 404:
        _, page = api(token,repo_path+'/pages','POST',{'build_type':'legacy','source':source})
    elif page.get('source') != source or page.get('build_type') != 'legacy':
        _, page = api(token,repo_path+'/pages','PUT',{'build_type':'legacy','source':source})
    print(json.dumps({'repository':repo['html_url'],'pagesUrl':page['html_url'],
                      'pagesStatus':page.get('status'),'commit':git('rev-parse','HEAD').stdout.strip()}))


if __name__ == '__main__':
    main()
