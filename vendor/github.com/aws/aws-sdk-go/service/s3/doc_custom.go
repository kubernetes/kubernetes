// Upload Managers
//
// The s3manager package's Uploader provides concurrent upload of content to S3
// by taking advantage of S3's Multipart APIs. The Uploader also supports both
// io.Reader for streaming uploads, and will also take advantage of io.ReadSeeker
// for optimizations if the Body satisfies that type. Once the Uploader instance
// is created you can call Upload concurrently from multiple goroutines safely.
//
//   // The session the S3 Uploader will use
//   sess := session.Must(session.NewSession())
//
//   // Create an uploader with the session and default options
//   uploader := s3manager.NewUploader(sess)
//
//   f, err  := os.Open(filename)
//   if err != nil {
//       return fmt.Errorf("failed to open file %q, %v", filename, err)
//   }
//
//   // Upload the file to S3.
//   result, err := uploader.Upload(&s3manager.UploadInput{
//       Bucket: aws.String(myBucket),
//       Key:    aws.String(myString),
//       Body:   f,
//   })
//   if err != nil {
//       return fmt.Errorf("failed to upload file, %v", err)
//   }
//   fmt.Printf("file uploaded to, %s\n", aws.StringValue(result.Location))
//
// See the s3manager package's Uploader type documentation for more information.
// https://docs.aws.amazon.com/sdk-for-go/api/service/s3/s3manager/#Uploader
//
// Download Manager
//
// The s3manager package's Downloader provides concurrently downloading of Objects
// from S3. The Downloader will write S3 Object content with an io.WriterAt.
// Once the Downloader instance is created you can call Upload concurrently from
// multiple goroutines safely.
//
//   // The session the S3 Downloader will use
//   sess := session.Must(session.NewSession())
//
//   // Create a downloader with the session and default options
//   downloader := s3manager.NewDownloader(sess)
//
//   // Create a file to write the S3 Object contents to.
//   f, err := os.Create(filename)
//   if err != nil {
//       return fmt.Errorf("failed to create file %q, %v", filename, err)
//   }
//
//   // Write the contents of S3 Object to the file
//   n, err := downloader.Download(f, &s3.GetObjectInput{
//       Bucket: aws.String(myBucket),
//       Key:    aws.String(myString),
//   })
//   if err != nil {
//       return fmt.Errorf("failed to upload file, %v", err)
//   }
//   fmt.Printf("file downloaded, %d bytes\n", n)
//
// See the s3manager package's Downloader type documentation for more information.
// https://docs.aws.amazon.com/sdk-for-go/api/service/s3/s3manager/#Downloader
//
// Get Bucket Region
//
// GetBucketRegion will attempt to get the region for a bucket using a region
// hint to determine which AWS partition to perform the query on. Use this utility
// to determine the region a bucket is in.
//
//   sess := session.Must(session.NewSession())
//
//   bucket := "my-bucket"
//   region, err := s3manager.GetBucketRegion(ctx, sess, bucket, "us-west-2")
//   if err != nil {
//       if aerr, ok := err.(awserr.Error); ok && aerr.Code() == "NotFound" {
//            fmt.Fprintf(os.Stderr, "unable to find bucket %s's region not found\n", bucket)
//       }
//       return err
//   }
//   fmt.Printf("Bucket %s is in %s region\n", bucket, region)
//
// See the s3manager package's GetBucketRegion function documentation for more information
// https://docs.aws.amazon.com/sdk-for-go/api/service/s3/s3manager/#GetBucketRegion
//
// S3 Crypto Client
//
// The s3crypto package provides the tools to upload and download encrypted
// content from S3. The Encryption and Decryption clients can be used concurrently
// once the client is created.
//
//    sess := session.Must(session.NewSession())
//
//    // Create the decryption client.
//    svc := s3crypto.NewDecryptionClient(sess)
//
//    // The object will be downloaded from S3 and decrypted locally. By metadata
//    // about the object's encryption will instruct the decryption client how
//    // decrypt the content of the object. By default KMS is used for keys.
//    result, err := svc.GetObject(&s3.GetObjectInput {
//        Bucket: aws.String(myBucket),
//        Key: aws.String(myKey),
//    })
//
// See the s3crypto package documentation for more information.
// https://docs.aws.amazon.com/sdk-for-go/api/service/s3/s3crypto/
//
package s3
