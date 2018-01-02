// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
Package vision provides a client for the Google Cloud Vision API.

Google Cloud Vision allows easy integration of vision detection features
into developer applications, including image labeling, face and landmark
detection, optical character recognition (OCR), and tagging of explicit
content. For more information about Cloud Vision, read the Google Cloud Vision API
Documentation at https://cloud.google.com/vision/docs.

Creating Images

The Cloud Vision API supports a variety of image file formats, including JPEG,
PNG8, PNG24, Animated GIF (first frame only), and RAW. See
https://cloud.google.com/vision/docs/image-best-practices#image_types for the
complete list of formats. Be aware that Cloud Vision sets upper limits on file
size as well as on the total combined size of all images in a request. Reducing
your file size can significantly improve throughput; however, be careful not to
reduce image quality in the process. See
https://cloud.google.com/vision/docs/image-best-practices#image_sizing for
current file size limits.

Creating an Image instance does not perform an API request.

Use NewImageFromReader to obtain an image from any io.Reader, such as an open file:

	f, err := os.Open("path/to/image.jpg")
	if err != nil { ... }
    defer f.Close()
	img, err := vision.NewImageFromReader(f)
	if err != nil { ... }

Use NewImageFromGCS to refer to an image in Google Cloud Storage:

	img := vision.NewImageFromGCS("gs://my-bucket/my-image.png")

Annotating Images

Client.Annotate is the most general method in the package. It can run multiple
detections on multiple images with a single API call.

To describe the detections you want to perform on an image, create an
AnnotateRequest and specify the maximum number of results to return for each
detection of interest. The exceptions are safe search and image properties,
where a boolean is used instead.

    resultSlice, err := client.Annotate(ctx, &vision.AnnotateRequest{
        Image:      img,
        MaxLogos:   5,
        MaxTexts:   100,
        SafeSearch: true,
    })
    if err != nil { ... }

You can pass as many AnnotateRequests as desired to client.Annotate. The return
value is a slice of an Annotations. Each Annotations value may contain an Error
along with one or more successful results. The failed detections will have a nil annotation.

    result := resultSlice[0]
    if result.Error != nil { ... } // some detections failed
    for _, logo := range result.Logos { ... }
    for _, text := range result.Texts { ... }
    if result.SafeSearch != nil { ... }

Other methods on Client run a single detection on a single image. For instance,
Client.DetectFaces will run face detection on the provided Image. These methods
return a single annotation of the appropriate type (for example, DetectFaces
returns a FaceAnnotation). The error return value incorporates both API call
errors and the detection errors stored in Annotations.Error, simplifying your
logic.

    faces, err := client.DetectFaces(ctx, 10) // maximum of 10 faces
    if err != nil { ... }

Here faces is a slice of FaceAnnotations. The Face field of each FaceAnnotation
provides easy access to the positions of facial features:

	fmt.Println(faces[0].Face.Nose.Tip)
	fmt.Println(faces[0].Face.Eyes.Left.Pupil)

This package is experimental and subject to API changes.
*/
package vision // import "cloud.google.com/go/vision"
