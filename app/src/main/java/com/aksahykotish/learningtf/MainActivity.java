package com.aksahykotish.learningtf;

import android.Manifest;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;

import com.aksahykotish.learningtf.ml.Twentytestingmodel;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.os.Debug;
import android.os.ParcelFileDescriptor;
import android.provider.MediaStore;
import android.view.View;


import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileDescriptor;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    Button camera, gallery;
    ImageView imageView;
    TextView textView;
    int imagesize = 180;

    int RESULT_LOAD_IMAGE = 123;
    int IMAGE_CAPTURE_CODE = 654;

    Uri image_uri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera = findViewById(R.id.camerabtn);
        gallery = findViewById(R.id.gallerybtn);
        imageView = findViewById(R.id.imageView);
        textView = findViewById(R.id.textView);


        if(camera != null) {
            camera.setOnClickListener(view -> {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {

                        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED || checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                                == PackageManager.PERMISSION_DENIED){
                            String[] permission = {Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE};
                            requestPermissions(permission, 112);
                        }
                    else{

                            ContentValues values = new ContentValues();
                            values.put(MediaStore.Images.Media.TITLE, "New Picture");
                            values.put(MediaStore.Images.Media.DESCRIPTION, "From the Camera");
                            image_uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
                            Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                            cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, image_uri);
                            startActivityForResult(cameraIntent, IMAGE_CAPTURE_CODE);
                        }


                }
            });
        }

        if(gallery != null) {
            gallery.setOnClickListener(view -> {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);
            });
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        Bitmap image = null;
        if(requestCode == 1)
        {
            Uri dat = data.getData();
            try {
                image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
            } catch (IOException e) {
                e.printStackTrace();
            }
            imageView.setImageBitmap(image);

        }
        else if (requestCode == IMAGE_CAPTURE_CODE && resultCode == RESULT_OK){
            imageView.setImageURI(image_uri);
            try {
                ParcelFileDescriptor parcelFileDescriptor =
                        getContentResolver().openFileDescriptor(image_uri, "r");
                FileDescriptor fileDescriptor = parcelFileDescriptor.getFileDescriptor();
                image = BitmapFactory.decodeFileDescriptor(fileDescriptor);

                parcelFileDescriptor.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        image = Bitmap.createScaledBitmap(image, imagesize, imagesize, false);
        classifyImage(image);
        super.onActivityResult(requestCode, resultCode, data);
    }

    private void classifyImage(Bitmap image) {
        try {
            Twentytestingmodel model = Twentytestingmodel.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 180, 180, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imagesize * imagesize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imagesize * imagesize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            int pixel = 0;
            for(int i=0; i<imagesize; i++ )
            {
                for(int j=0; j<imagesize; j++)
                {
                    int val = intValues[pixel++]; //RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat((val& 0xFF) * (1.f / 1));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Twentytestingmodel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            //Find the index of  the class with biggest confidence

            int maxPos = 0;
            float maxConfidence = 0;
            for(int i=0; i<confidences.length; i++)
            {
                if(confidences[i] > maxConfidence)
                {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            String[] classes  = {"zapiz-0.25-tablet-57542", "zapiz-0.5-tablet-57643", "zedex-cough-syrup-67280", "zenflox-oz-tablet-393314", "zenflox-uti-tablet-150320", "zentel-chewable-tablet-137773", "zentel-oral-suspension-41573", "zerodol-mr-tablet-67303", "zerodol-p-tablet-67304", "zerodol-sp-tablet-67307", "zerodol-tablet-191575", "zerodol-th-4-tablet-122546", "zifi-200-tablet-51344", "zifi-cv-200-tablet-161543", "zifi-o-tablet-116738", "zocef-500-tablet-72982", "zolfresh-10mg-tablet-328296", "zolfresh-5-tablet-328242", "zyloric-tablet-41865", "zytee-rb-gel-135783"};

            textView.setText(classes[maxPos]);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }
}