import unittest
from PIL import Image
from scripts.merge_v12_dataset import square_two_pixel_fix

class TwoPixelCropTests(unittest.TestCase):
    def test_wide_image_trims_one_pixel_from_each_side(self):
        im=Image.new('RGB',(1254,1252),'white');im.putpixel((1,0),(10,20,30))
        out,changed=square_two_pixel_fix(im)
        self.assertTrue(changed);self.assertEqual(out.size,(1252,1252))
        self.assertEqual(out.getpixel((0,0)),(10,20,30));self.assertEqual(im.size,(1254,1252))
    def test_tall_image_trims_one_row_from_each_end(self):
        out,changed=square_two_pixel_fix(Image.new('RGB',(1252,1254)))
        self.assertTrue(changed);self.assertEqual(out.size,(1252,1252))
    def test_other_aspect_ratios_are_not_cropped(self):
        im=Image.new('RGB',(1200,800));out,changed=square_two_pixel_fix(im)
        self.assertFalse(changed);self.assertIs(out,im)

if __name__=='__main__':unittest.main()
