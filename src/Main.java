import java.util.ArrayList;
import java.util.List;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;

public class Main {

    public static boolean isRunning = true;

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Imshow im = new Imshow("Video Preview");

        im.Window.setResizable(true);
        
        Mat image = new Mat();
        VideoCapture vcam = new VideoCapture(0);
        
        im.Window.addWindowListener(new WindowAdapter()
        {
            public void windowClosing(WindowEvent e)
            {
                isRunning = false;
                vcam.release(); // required to let java exit properly
            }
        });

        // loop until VideoCamera is Available
        while (vcam.isOpened() == false)
            ;

        // loop until image frames are not empty
        while (image.empty()) {
            vcam.read(image);
        }
        
        CascadeClassifier faceCascade = new CascadeClassifier("data/haarcascades/haarcascade_frontalface_alt.xml");
        CascadeClassifier eyeCascade = new CascadeClassifier("data/haarcascades/haarcascade_eye.xml");
        
        Mat grayImage = new Mat();
        MatOfRect faces = new MatOfRect();
        int minFaceSize = 0;
        int minEyeSize = 0;
        Mat hatimg = Imgcodecs.imread("data/hat2.png");
        float hatAspect = hatimg.width() / (float)hatimg.height();
        
        while (isRunning) {
            vcam.read(image);
                      
            Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGR2GRAY);
            Imgproc.equalizeHist(grayImage, grayImage);
            
            if (minFaceSize == 0)
            {
                int height = grayImage.rows();
                minFaceSize = Math.round(height * 0.2f);
                minEyeSize = Math.max(Math.round(minFaceSize * 0.1f), 20);
            }

            faceCascade.detectMultiScale(grayImage, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE, new Size(minFaceSize, minFaceSize), new Size());
            
            Rect[] facesArray = faces.toArray();
            for (int i = 0; i < facesArray.length; i++) {
                
                Point faceTl = facesArray[i].tl();
                Point faceBr = facesArray[i].br();
                Mat faceMat = grayImage.submat(facesArray[i]);
                
                Imgproc.rectangle(image, faceTl, faceBr, new Scalar(0, 255, 0, 255), 3);
                
                MatOfRect eyes = new MatOfRect();
                eyeCascade.detectMultiScale(faceMat, eyes, 1.1, 10, 0 | Objdetect.CASCADE_SCALE_IMAGE, new Size(minEyeSize, minEyeSize), new Size());
                
                Rect[] eyesArray = eyes.toArray();
                for (int j = 0; j < eyesArray.length; j++) {
                    
                    Point eyeRelTl = eyesArray[j].tl();
                    Point eyeRelBr = eyesArray[j].br();
                    
                    Point eyeAbsTl = new Point(eyeRelTl.x + faceTl.x, eyeRelTl.y + faceTl.y);  
                    Point eyeAbsBr = new Point(eyeRelBr.x + faceTl.x, eyeRelBr.y + faceTl.y);  
                    
                    Imgproc.rectangle(image, eyeAbsTl, eyeAbsBr, new Scalar(255, 0, 0, 255), 2);
                }
                
                
                Rect roi = facesArray[i].clone();
                Size hatSize = new Size(roi.width * 0.8, roi.width * 0.8 / hatAspect);
                Mat hat = new Mat();
                Imgproc.resize(hatimg, hat, hatSize);
                
                roi.x = (int) Math.max(roi.x + 0.5 * (roi.width - hatSize.width), 0);
                roi.y = (int) Math.max(roi.y - hatSize.height, 0);
                roi.height = (int) Math.min(hatSize.height, image.height() - roi.y);
                roi.width  = (int) Math.min(hatSize.width, image.width() - roi.x);
                hat.copyTo(image.submat(roi));
            }
            
            //Mat dft = applyDFT(grayImage);
            //Mat restored = applyInverseDFT(dft, CvType.CV_8U);
            
            im.showImage(image);
        }
        vcam.release();
        System.exit(0);
    }
    
    /**
     * Applies Discrete Fourier Transform to {@code image}
     * IMPORTANT: Make sure image is in grayscale (CV_8U) format.
     * @return new complex (two-channel float32) {@link Mat} containing the DFT result
     */
    public static Mat applyDFT(Mat image) {
        
        if (image.type() != CvType.CV_8U)
            throw new IllegalArgumentException("DFT only supports CvType.CV_8U format");
        
        // pad the image for optimal DFT 
        Mat padded = new Mat();
        int addPixelRows = Core.getOptimalDFTSize(image.rows());
        int addPixelCols = Core.getOptimalDFTSize(image.cols());
        Core.copyMakeBorder(image, padded, 0, addPixelRows - image.rows(), 0, addPixelCols - image.cols(), Core.BORDER_CONSTANT, Scalar.all(0));
        
        List<Mat> planes = new ArrayList<Mat>();
        Mat complexImage = new Mat();
        
        // convert to float format, expand into new two-channel image (complexImage) to hold DFT result
        padded.convertTo(padded, CvType.CV_32F);
        planes.add(padded);
        planes.add(Mat.zeros(padded.size(), CvType.CV_32F));
        Core.merge(planes, complexImage);
        
      
        // apply DFT 
        Core.dft(complexImage, complexImage);
        
        // compute magnitude
//        planes.clear();
//        Core.split(complexImage, planes);
//        Mat mag = new Mat();
//        Core.magnitude(planes.get(0), planes.get(1), mag);
        
        return complexImage;
    }
    
    /**
     * Applies Inverse Discrete Fourier Transform to {@code complexImage}
     * @return new {@link Mat} converted to {@link CvType} {@code outputType}
     */
    public static Mat applyInverseDFT(Mat complexImage, int outputType) {
        
        List<Mat> planes = new ArrayList<Mat>();
        Mat restoredImage = new Mat();
       
        Core.idft(complexImage, restoredImage);
        Core.split(restoredImage, planes);
        Core.normalize(planes.get(0), restoredImage, 0, 255, Core.NORM_MINMAX);
        restoredImage.convertTo(restoredImage, outputType);
        
        return restoredImage;
    }
 }