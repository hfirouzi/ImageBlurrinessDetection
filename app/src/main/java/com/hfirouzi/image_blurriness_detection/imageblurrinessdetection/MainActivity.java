package com.hfirouzi.image_blurriness_detection.imageblurrinessdetection;

import android.Manifest;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.imgproc.Imgproc;

import java.io.File;

/**
 * Created by Hadi Firouzi, hfirouzi@gmail.com
 * This code is under Apache License, 2.0 (Apache-2.0)
 */

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MAIN_ACTIVITY::TAG";
    private static final String TEMP_PHOTO_PATH = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM) + "/test_photo.jpg";
    private static final int REQUEST_IMAGE_CAPTURE = 1001;
    private static final double MIN_IMAGE_BLURRINESS = 10;
    private static final int IMAGE_PATCH_SIZE = 200;

    // initialize OpenCV
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.d(TAG, "OpenCV loaded successfully");
                }
                break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button takePhoto = (Button) findViewById(R.id.take_photo);
        takePhoto.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                launchCamera();
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

    }

    private void launchCamera() {
        int permissionCheck = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA);
        if (permissionCheck == PackageManager.PERMISSION_GRANTED) {
            Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, Uri.fromFile(new File(TEMP_PHOTO_PATH)));
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
            } else {
                Toast.makeText(this, "Could not find Camera app!", Toast.LENGTH_SHORT).show();
            }
        } else {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    1);
        }

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == REQUEST_IMAGE_CAPTURE) {
            if (resultCode == RESULT_OK) {
                Bitmap bitmap = BitmapFactory.decodeFile(TEMP_PHOTO_PATH);
                ImageView imageView = (ImageView) findViewById(R.id.imageView);
                imageView.setImageBitmap(bitmap);
                double blurriness = calculateBlurriness(bitmap);
                if (blurriness < MIN_IMAGE_BLURRINESS) {
                    int blurrinessPercentage = (int) ((MIN_IMAGE_BLURRINESS - blurriness) / MIN_IMAGE_BLURRINESS * 100);
                    new AlertDialog.Builder(this)
                            .setMessage("Image blurriness is % " + blurrinessPercentage + ", please take another photo!")
                            .setNeutralButton("Ok", null)
                            .setTitle("Blurry Image")
                            .create()
                            .show();

                } else {
                    Toast.makeText(this, "Good job! Photo is not blurry", Toast.LENGTH_LONG).show();
                }
            }
        }

    }

    private double calculateBlurriness(Bitmap bitmap) {
        Mat image = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_8U);
        Mat imageGray = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_8UC1);
        Utils.bitmapToMat(bitmap, image);
        Imgproc.cvtColor(image, imageGray, Imgproc.COLOR_RGBA2GRAY);

        Mat imageLaplac = Mat.zeros(imageGray.rows(), imageGray.cols(), CvType.CV_16SC1);
        Imgproc.Laplacian(imageGray, imageLaplac, CvType.CV_16S);
        Mat imageLaplacAbs = Mat.zeros(imageLaplac.rows(), imageLaplac.cols(), imageLaplac.type());
        Core.convertScaleAbs(imageLaplac, imageLaplacAbs);

        Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(imageLaplacAbs);

        int radius = imageLaplacAbs.width() > 2*IMAGE_PATCH_SIZE ? IMAGE_PATCH_SIZE : (int) (imageLaplacAbs.width() / 2.0);
        int x1 = minMaxLocResult.maxLoc.x>radius ? (int) (minMaxLocResult.maxLoc.x - radius) : 0;
        int x2 = x1 + 2*radius;
        if (x2 >= imageLaplacAbs.width()) {
            x2 = imageLaplacAbs.width() - 1;
            x1 = x2 - 2*radius;
        }
        int y1 = minMaxLocResult.maxLoc.y>radius ? (int) (minMaxLocResult.maxLoc.y - radius) : 0;
        int y2 = y1 + 2*radius;
        if (y2 >= imageLaplacAbs.height()) {
            y2 = imageLaplacAbs.height() - 1;
            y1 = y2 - 2*radius;
        }

        Mat subImageLaplacAbs = imageLaplacAbs.submat(y1, y2, x1, x2);

        MatOfDouble mu = new MatOfDouble();
        MatOfDouble sigma = new MatOfDouble();
        Core.meanStdDev(subImageLaplacAbs, mu, sigma);
        return sigma.get(0,0)[0];
    }

}
