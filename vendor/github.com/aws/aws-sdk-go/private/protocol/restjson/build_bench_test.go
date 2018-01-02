// +build bench

package restjson_test

import (
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/endpoints"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/private/protocol/restjson"
	"github.com/aws/aws-sdk-go/service/elastictranscoder"
)

var (
	elastictranscoderSvc *elastictranscoder.ElasticTranscoder
)

func TestMain(m *testing.M) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	sess := session.Must(session.NewSession(&aws.Config{
		Credentials:      credentials.NewStaticCredentials("Key", "Secret", "Token"),
		Endpoint:         aws.String(server.URL),
		S3ForcePathStyle: aws.Bool(true),
		DisableSSL:       aws.Bool(true),
		Region:           aws.String(endpoints.UsWest2RegionID),
	}))
	elastictranscoderSvc = elastictranscoder.New(sess)

	c := m.Run()
	server.Close()
	os.Exit(c)
}

func BenchmarkRESTJSONBuild_Complex_ETCCreateJob(b *testing.B) {
	params := elastictranscoderCreateJobInput()

	benchRESTJSONBuild(b, func() *request.Request {
		req, _ := elastictranscoderSvc.CreateJobRequest(params)
		return req
	})
}

func BenchmarkRESTJSONBuild_Simple_ETCListJobsByPipeline(b *testing.B) {
	params := elastictranscoderListJobsByPipeline()

	benchRESTJSONBuild(b, func() *request.Request {
		req, _ := elastictranscoderSvc.ListJobsByPipelineRequest(params)
		return req
	})
}

func BenchmarkRESTJSONRequest_Complex_CFCreateJob(b *testing.B) {
	benchRESTJSONRequest(b, func() *request.Request {
		req, _ := elastictranscoderSvc.CreateJobRequest(elastictranscoderCreateJobInput())
		return req
	})
}

func BenchmarkRESTJSONRequest_Simple_ETCListJobsByPipeline(b *testing.B) {
	benchRESTJSONRequest(b, func() *request.Request {
		req, _ := elastictranscoderSvc.ListJobsByPipelineRequest(elastictranscoderListJobsByPipeline())
		return req
	})
}

func benchRESTJSONBuild(b *testing.B, reqFn func() *request.Request) {
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		req := reqFn()
		restjson.Build(req)
		if req.Error != nil {
			b.Fatal("Unexpected error", req.Error)
		}
	}
}

func benchRESTJSONRequest(b *testing.B, reqFn func() *request.Request) {
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		err := reqFn().Send()
		if err != nil {
			b.Fatal("Unexpected error", err)
		}
	}
}

func elastictranscoderListJobsByPipeline() *elastictranscoder.ListJobsByPipelineInput {
	return &elastictranscoder.ListJobsByPipelineInput{
		PipelineId: aws.String("Id"), // Required
		Ascending:  aws.String("Ascending"),
		PageToken:  aws.String("Id"),
	}
}

func elastictranscoderCreateJobInput() *elastictranscoder.CreateJobInput {
	return &elastictranscoder.CreateJobInput{
		Input: &elastictranscoder.JobInput{ // Required
			AspectRatio: aws.String("AspectRatio"),
			Container:   aws.String("JobContainer"),
			DetectedProperties: &elastictranscoder.DetectedProperties{
				DurationMillis: aws.Int64(1),
				FileSize:       aws.Int64(1),
				FrameRate:      aws.String("FloatString"),
				Height:         aws.Int64(1),
				Width:          aws.Int64(1),
			},
			Encryption: &elastictranscoder.Encryption{
				InitializationVector: aws.String("ZeroTo255String"),
				Key:                  aws.String("Base64EncodedString"),
				KeyMd5:               aws.String("Base64EncodedString"),
				Mode:                 aws.String("EncryptionMode"),
			},
			FrameRate:  aws.String("FrameRate"),
			Interlaced: aws.String("Interlaced"),
			Key:        aws.String("Key"),
			Resolution: aws.String("Resolution"),
		},
		PipelineId: aws.String("Id"), // Required
		Output: &elastictranscoder.CreateJobOutput{
			AlbumArt: &elastictranscoder.JobAlbumArt{
				Artwork: []*elastictranscoder.Artwork{
					{ // Required
						AlbumArtFormat: aws.String("JpgOrPng"),
						Encryption: &elastictranscoder.Encryption{
							InitializationVector: aws.String("ZeroTo255String"),
							Key:                  aws.String("Base64EncodedString"),
							KeyMd5:               aws.String("Base64EncodedString"),
							Mode:                 aws.String("EncryptionMode"),
						},
						InputKey:      aws.String("WatermarkKey"),
						MaxHeight:     aws.String("DigitsOrAuto"),
						MaxWidth:      aws.String("DigitsOrAuto"),
						PaddingPolicy: aws.String("PaddingPolicy"),
						SizingPolicy:  aws.String("SizingPolicy"),
					},
					// More values...
				},
				MergePolicy: aws.String("MergePolicy"),
			},
			Captions: &elastictranscoder.Captions{
				CaptionFormats: []*elastictranscoder.CaptionFormat{
					{ // Required
						Encryption: &elastictranscoder.Encryption{
							InitializationVector: aws.String("ZeroTo255String"),
							Key:                  aws.String("Base64EncodedString"),
							KeyMd5:               aws.String("Base64EncodedString"),
							Mode:                 aws.String("EncryptionMode"),
						},
						Format:  aws.String("CaptionFormatFormat"),
						Pattern: aws.String("CaptionFormatPattern"),
					},
					// More values...
				},
				CaptionSources: []*elastictranscoder.CaptionSource{
					{ // Required
						Encryption: &elastictranscoder.Encryption{
							InitializationVector: aws.String("ZeroTo255String"),
							Key:                  aws.String("Base64EncodedString"),
							KeyMd5:               aws.String("Base64EncodedString"),
							Mode:                 aws.String("EncryptionMode"),
						},
						Key:        aws.String("Key"),
						Label:      aws.String("Name"),
						Language:   aws.String("Key"),
						TimeOffset: aws.String("TimeOffset"),
					},
					// More values...
				},
				MergePolicy: aws.String("CaptionMergePolicy"),
			},
			Composition: []*elastictranscoder.Clip{
				{ // Required
					TimeSpan: &elastictranscoder.TimeSpan{
						Duration:  aws.String("Time"),
						StartTime: aws.String("Time"),
					},
				},
				// More values...
			},
			Encryption: &elastictranscoder.Encryption{
				InitializationVector: aws.String("ZeroTo255String"),
				Key:                  aws.String("Base64EncodedString"),
				KeyMd5:               aws.String("Base64EncodedString"),
				Mode:                 aws.String("EncryptionMode"),
			},
			Key:             aws.String("Key"),
			PresetId:        aws.String("Id"),
			Rotate:          aws.String("Rotate"),
			SegmentDuration: aws.String("FloatString"),
			ThumbnailEncryption: &elastictranscoder.Encryption{
				InitializationVector: aws.String("ZeroTo255String"),
				Key:                  aws.String("Base64EncodedString"),
				KeyMd5:               aws.String("Base64EncodedString"),
				Mode:                 aws.String("EncryptionMode"),
			},
			ThumbnailPattern: aws.String("ThumbnailPattern"),
			Watermarks: []*elastictranscoder.JobWatermark{
				{ // Required
					Encryption: &elastictranscoder.Encryption{
						InitializationVector: aws.String("ZeroTo255String"),
						Key:                  aws.String("Base64EncodedString"),
						KeyMd5:               aws.String("Base64EncodedString"),
						Mode:                 aws.String("EncryptionMode"),
					},
					InputKey:          aws.String("WatermarkKey"),
					PresetWatermarkId: aws.String("PresetWatermarkId"),
				},
				// More values...
			},
		},
		OutputKeyPrefix: aws.String("Key"),
		Outputs: []*elastictranscoder.CreateJobOutput{
			{ // Required
				AlbumArt: &elastictranscoder.JobAlbumArt{
					Artwork: []*elastictranscoder.Artwork{
						{ // Required
							AlbumArtFormat: aws.String("JpgOrPng"),
							Encryption: &elastictranscoder.Encryption{
								InitializationVector: aws.String("ZeroTo255String"),
								Key:                  aws.String("Base64EncodedString"),
								KeyMd5:               aws.String("Base64EncodedString"),
								Mode:                 aws.String("EncryptionMode"),
							},
							InputKey:      aws.String("WatermarkKey"),
							MaxHeight:     aws.String("DigitsOrAuto"),
							MaxWidth:      aws.String("DigitsOrAuto"),
							PaddingPolicy: aws.String("PaddingPolicy"),
							SizingPolicy:  aws.String("SizingPolicy"),
						},
						// More values...
					},
					MergePolicy: aws.String("MergePolicy"),
				},
				Captions: &elastictranscoder.Captions{
					CaptionFormats: []*elastictranscoder.CaptionFormat{
						{ // Required
							Encryption: &elastictranscoder.Encryption{
								InitializationVector: aws.String("ZeroTo255String"),
								Key:                  aws.String("Base64EncodedString"),
								KeyMd5:               aws.String("Base64EncodedString"),
								Mode:                 aws.String("EncryptionMode"),
							},
							Format:  aws.String("CaptionFormatFormat"),
							Pattern: aws.String("CaptionFormatPattern"),
						},
						// More values...
					},
					CaptionSources: []*elastictranscoder.CaptionSource{
						{ // Required
							Encryption: &elastictranscoder.Encryption{
								InitializationVector: aws.String("ZeroTo255String"),
								Key:                  aws.String("Base64EncodedString"),
								KeyMd5:               aws.String("Base64EncodedString"),
								Mode:                 aws.String("EncryptionMode"),
							},
							Key:        aws.String("Key"),
							Label:      aws.String("Name"),
							Language:   aws.String("Key"),
							TimeOffset: aws.String("TimeOffset"),
						},
						// More values...
					},
					MergePolicy: aws.String("CaptionMergePolicy"),
				},
				Composition: []*elastictranscoder.Clip{
					{ // Required
						TimeSpan: &elastictranscoder.TimeSpan{
							Duration:  aws.String("Time"),
							StartTime: aws.String("Time"),
						},
					},
					// More values...
				},
				Encryption: &elastictranscoder.Encryption{
					InitializationVector: aws.String("ZeroTo255String"),
					Key:                  aws.String("Base64EncodedString"),
					KeyMd5:               aws.String("Base64EncodedString"),
					Mode:                 aws.String("EncryptionMode"),
				},
				Key:             aws.String("Key"),
				PresetId:        aws.String("Id"),
				Rotate:          aws.String("Rotate"),
				SegmentDuration: aws.String("FloatString"),
				ThumbnailEncryption: &elastictranscoder.Encryption{
					InitializationVector: aws.String("ZeroTo255String"),
					Key:                  aws.String("Base64EncodedString"),
					KeyMd5:               aws.String("Base64EncodedString"),
					Mode:                 aws.String("EncryptionMode"),
				},
				ThumbnailPattern: aws.String("ThumbnailPattern"),
				Watermarks: []*elastictranscoder.JobWatermark{
					{ // Required
						Encryption: &elastictranscoder.Encryption{
							InitializationVector: aws.String("ZeroTo255String"),
							Key:                  aws.String("Base64EncodedString"),
							KeyMd5:               aws.String("Base64EncodedString"),
							Mode:                 aws.String("EncryptionMode"),
						},
						InputKey:          aws.String("WatermarkKey"),
						PresetWatermarkId: aws.String("PresetWatermarkId"),
					},
					// More values...
				},
			},
			// More values...
		},
		Playlists: []*elastictranscoder.CreateJobPlaylist{
			{ // Required
				Format: aws.String("PlaylistFormat"),
				HlsContentProtection: &elastictranscoder.HlsContentProtection{
					InitializationVector:  aws.String("ZeroTo255String"),
					Key:                   aws.String("Base64EncodedString"),
					KeyMd5:                aws.String("Base64EncodedString"),
					KeyStoragePolicy:      aws.String("KeyStoragePolicy"),
					LicenseAcquisitionUrl: aws.String("ZeroTo512String"),
					Method:                aws.String("HlsContentProtectionMethod"),
				},
				Name: aws.String("Filename"),
				OutputKeys: []*string{
					aws.String("Key"), // Required
					// More values...
				},
				PlayReadyDrm: &elastictranscoder.PlayReadyDrm{
					Format:                aws.String("PlayReadyDrmFormatString"),
					InitializationVector:  aws.String("ZeroTo255String"),
					Key:                   aws.String("NonEmptyBase64EncodedString"),
					KeyId:                 aws.String("KeyIdGuid"),
					KeyMd5:                aws.String("NonEmptyBase64EncodedString"),
					LicenseAcquisitionUrl: aws.String("OneTo512String"),
				},
			},
			// More values...
		},
		UserMetadata: map[string]*string{
			"Key": aws.String("String"), // Required
			// More values...
		},
	}
}
