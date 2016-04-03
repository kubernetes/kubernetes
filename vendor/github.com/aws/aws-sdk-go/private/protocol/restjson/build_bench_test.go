// +build bench

package restjson_test

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting"
	"github.com/aws/aws-sdk-go/private/protocol/rest"
	"github.com/aws/aws-sdk-go/private/protocol/restjson"
	"github.com/aws/aws-sdk-go/service/elastictranscoder"
)

func BenchmarkRESTJSONBuild_Complex_elastictranscoderCreateJobInput(b *testing.B) {
	svc := awstesting.NewClient()
	svc.ServiceName = "elastictranscoder"
	svc.APIVersion = "2012-09-25"

	for i := 0; i < b.N; i++ {
		r := svc.NewRequest(&request.Operation{Name: "CreateJobInput"}, restjsonBuildParms, nil)
		restjson.Build(r)
		if r.Error != nil {
			b.Fatal("Unexpected error", r.Error)
		}
	}
}

func BenchmarkRESTBuild_Complex_elastictranscoderCreateJobInput(b *testing.B) {
	svc := awstesting.NewClient()
	svc.ServiceName = "elastictranscoder"
	svc.APIVersion = "2012-09-25"

	for i := 0; i < b.N; i++ {
		r := svc.NewRequest(&request.Operation{Name: "CreateJobInput"}, restjsonBuildParms, nil)
		rest.Build(r)
		if r.Error != nil {
			b.Fatal("Unexpected error", r.Error)
		}
	}
}

func BenchmarkEncodingJSONMarshal_Complex_elastictranscoderCreateJobInput(b *testing.B) {
	params := restjsonBuildParms

	for i := 0; i < b.N; i++ {
		buf := &bytes.Buffer{}
		encoder := json.NewEncoder(buf)
		if err := encoder.Encode(params); err != nil {
			b.Fatal("Unexpected error", err)
		}
	}
}

func BenchmarkRESTJSONBuild_Simple_elastictranscoderListJobsByPipeline(b *testing.B) {
	svc := awstesting.NewClient()
	svc.ServiceName = "elastictranscoder"
	svc.APIVersion = "2012-09-25"

	params := &elastictranscoder.ListJobsByPipelineInput{
		PipelineId: aws.String("Id"), // Required
		Ascending:  aws.String("Ascending"),
		PageToken:  aws.String("Id"),
	}

	for i := 0; i < b.N; i++ {
		r := svc.NewRequest(&request.Operation{Name: "ListJobsByPipeline"}, params, nil)
		restjson.Build(r)
		if r.Error != nil {
			b.Fatal("Unexpected error", r.Error)
		}
	}
}

func BenchmarkRESTBuild_Simple_elastictranscoderListJobsByPipeline(b *testing.B) {
	svc := awstesting.NewClient()
	svc.ServiceName = "elastictranscoder"
	svc.APIVersion = "2012-09-25"

	params := &elastictranscoder.ListJobsByPipelineInput{
		PipelineId: aws.String("Id"), // Required
		Ascending:  aws.String("Ascending"),
		PageToken:  aws.String("Id"),
	}

	for i := 0; i < b.N; i++ {
		r := svc.NewRequest(&request.Operation{Name: "ListJobsByPipeline"}, params, nil)
		rest.Build(r)
		if r.Error != nil {
			b.Fatal("Unexpected error", r.Error)
		}
	}
}

func BenchmarkEncodingJSONMarshal_Simple_elastictranscoderListJobsByPipeline(b *testing.B) {
	params := &elastictranscoder.ListJobsByPipelineInput{
		PipelineId: aws.String("Id"), // Required
		Ascending:  aws.String("Ascending"),
		PageToken:  aws.String("Id"),
	}

	for i := 0; i < b.N; i++ {
		buf := &bytes.Buffer{}
		encoder := json.NewEncoder(buf)
		if err := encoder.Encode(params); err != nil {
			b.Fatal("Unexpected error", err)
		}
	}
}

var restjsonBuildParms = &elastictranscoder.CreateJobInput{
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
