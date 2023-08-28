using OpenCvSharp;
using System.Collections;

namespace ImageProcessing
{
    public class ImageData
    {
        public Hashtable detectFace(string imageFileWithPath, string cascadeClassifierFileWithPath, int percentageDisplacement)
        {
            Hashtable imageData = new Hashtable();

            using var image = new Mat(imageFileWithPath);

            using var gray = new Mat();
            Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);
            var faceCascade = new CascadeClassifier(cascadeClassifierFileWithPath); // Load the Haar Cascade XML file

            var faces = faceCascade.DetectMultiScale(gray, 1.1, 5, 0, new Size(30, 30));
            int result = faces.Length;
            Console.WriteLine("Number of faces detected: " + result);

            int ImageWidth = image.Cols;
            int ImageHeight = image.Rows;

            int centerX = image.Cols / 2;
            int centerY = image.Rows / 2;


            if (result == 1)
            {
                foreach (var face in faces)
                {

                    int radius = centerX * percentageDisplacement / 100;

                    Cv2.Circle(image, new Point(centerX, centerY), radius, Scalar.Red, 2); // Circle of CenterImage
                    Cv2.Circle(image, new Point(centerX, centerY), 1, Scalar.Red, 2); // center point of Circle 40percent

                    int CenterFaceX = Math.Max(face.X + (face.Width / 2), 0);
                    int CenterFaceY = Math.Max(face.Y + (face.Height / 2), 0);

                    Cv2.Circle(image, new Point(CenterFaceX, CenterFaceY), 1, Scalar.Blue, 2); // Center Point of Face

                    double EqX = Math.Pow((CenterFaceX - centerX), 2);
                    double EqY = Math.Pow((CenterFaceY - centerY), 2);

                    double EqRadius = Math.Pow(radius, 2);

                    double resultant = ((EqX + EqY) - EqRadius); // inside (-ve ) on the circle (0) or +ve if outside

                    imageData.Add("Resultant", resultant);


                    double angleRad = Math.Atan2(CenterFaceY - centerY, CenterFaceX - centerX);
                    double angleDeg = angleRad * (180.0 / Math.PI);

                    imageData.Add("Degrees", angleDeg);

                }
                return imageData;
            }

                return new Hashtable();

        }

    }
}