// +build integration

package transcribestreamingservice

import (
	"bytes"
	"context"
	"encoding/base64"
	"flag"
	"io"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting/integration"
)

var (
	audioFilename   string
	audioFormat     string
	audioLang       string
	audioSampleRate int
	audioFrameSize  int
	withDebug       bool
)

func init() {
	flag.BoolVar(&withDebug, "debug", false, "Include debug logging with test.")
	flag.StringVar(&audioFilename, "audio-file", "", "Audio file filename to perform test with.")
	flag.StringVar(&audioLang, "audio-lang", LanguageCodeEnUs, "Language of audio speech.")
	flag.StringVar(&audioFormat, "audio-format", MediaEncodingPcm, "Format of audio.")
	flag.IntVar(&audioSampleRate, "audio-sample", 16000, "Sample rate of the audio.")
	flag.IntVar(&audioFrameSize, "audio-frame", 15*1024, "Size of frames of audio uploaded.")
}

func TestInteg_StartStreamTranscription(t *testing.T) {
	var audio io.Reader
	if len(audioFilename) != 0 {
		audioFile, err := os.Open(audioFilename)
		if err != nil {
			t.Fatalf("expect to open file, %v", err)
		}
		defer audioFile.Close()
		audio = audioFile
	} else {
		b, err := base64.StdEncoding.DecodeString(
			`UklGRjzxPQBXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YVTwPQAAAAAAAAAAAAAAAAD//wIA/f8EAA==`,
		)
		if err != nil {
			t.Fatalf("expect decode audio bytes, %v", err)
		}
		audio = bytes.NewReader(b)
	}

	sess := integration.SessionWithDefaultRegion("us-west-2")
	var cfgs []*aws.Config
	if withDebug {
		cfgs = append(cfgs, &aws.Config{
			Logger:   t,
			LogLevel: aws.LogLevel(aws.LogDebugWithEventStreamBody),
		})
	}

	client := New(sess, cfgs...)
	resp, err := client.StartStreamTranscription(&StartStreamTranscriptionInput{
		LanguageCode:         aws.String(audioLang),
		MediaEncoding:        aws.String(audioFormat),
		MediaSampleRateHertz: aws.Int64(int64(audioSampleRate)),
	})
	if err != nil {
		t.Fatalf("failed to start streaming, %v", err)
	}
	stream := resp.GetStream()
	defer stream.Close()

	go StreamAudioFromReader(context.Background(), stream.Writer, audioFrameSize, audio)

	for event := range stream.Events() {
		switch e := event.(type) {
		case *TranscriptEvent:
			t.Logf("got event, %v results", len(e.Transcript.Results))
			for _, res := range e.Transcript.Results {
				for _, alt := range res.Alternatives {
					t.Logf("* %s", aws.StringValue(alt.Transcript))
				}
			}
		default:
			t.Fatalf("unexpected event, %T", event)
		}
	}

	if err := stream.Err(); err != nil {
		t.Fatalf("expect no error from stream, got %v", err)
	}
}

func TestInteg_StartStreamTranscription_contextClose(t *testing.T) {
	b, err := base64.StdEncoding.DecodeString(
		`UklGRjzxPQBXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YVTwPQAAAAAAAAAAAAAAAAD//wIA/f8EAA==`,
	)
	if err != nil {
		t.Fatalf("expect decode audio bytes, %v", err)
	}
	audio := bytes.NewReader(b)

	sess := integration.SessionWithDefaultRegion("us-west-2")
	var cfgs []*aws.Config

	client := New(sess, cfgs...)
	resp, err := client.StartStreamTranscription(&StartStreamTranscriptionInput{
		LanguageCode:         aws.String(LanguageCodeEnUs),
		MediaEncoding:        aws.String(MediaEncodingPcm),
		MediaSampleRateHertz: aws.Int64(16000),
	})
	if err != nil {
		t.Fatalf("failed to start streaming, %v", err)
	}
	stream := resp.GetStream()
	defer stream.Close()

	ctx, cancelFn := context.WithCancel(context.Background())
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		err := StreamAudioFromReader(ctx, stream.Writer, audioFrameSize, audio)
		if err == nil {
			t.Errorf("expect error")
		}
		if e, a := "context canceled", err.Error(); !strings.Contains(a, e) {
			t.Errorf("expect %q error in %q", e, a)
		}
		wg.Done()
	}()

	cancelFn()

Loop:
	for {
		select {
		case <-ctx.Done():
			break Loop
		case event, ok := <-stream.Events():
			if !ok {
				break Loop
			}
			switch e := event.(type) {
			case *TranscriptEvent:
				t.Logf("got event, %v results", len(e.Transcript.Results))
				for _, res := range e.Transcript.Results {
					for _, alt := range res.Alternatives {
						t.Logf("* %s", aws.StringValue(alt.Transcript))
					}
				}
			default:
				t.Fatalf("unexpected event, %T", event)
			}
		}
	}

	wg.Wait()

	if err := stream.Err(); err != nil {
		t.Fatalf("expect no error from stream, got %v", err)
	}
}
