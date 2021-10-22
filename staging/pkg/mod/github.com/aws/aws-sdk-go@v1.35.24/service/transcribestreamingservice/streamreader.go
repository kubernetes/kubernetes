package transcribestreamingservice

import (
	"fmt"
	"io"

	"github.com/aws/aws-sdk-go/aws"
)

// StreamAudioFromReader will stream bytes from the provided input io.Reader to
// the AudioStreamWriter in chunks of frameSize in length. Returns an error if
// streaming to AudioStreamWriter fails.
func StreamAudioFromReader(ctx aws.Context, stream AudioStreamWriter, frameSize int, input io.Reader) (err error) {
	defer func() {
		if closeErr := stream.Close(); closeErr != nil && err == nil {
			err = fmt.Errorf("failed to close stream, %v", closeErr)
		}
	}()

	frame := make([]byte, frameSize)
	for {
		var n int
		n, err = input.Read(frame)
		if n > 0 {
			err = stream.Send(ctx, &AudioEvent{
				AudioChunk: frame[:n],
			})
			if err != nil {
				return fmt.Errorf("failed to send audio event, %v", err)
			}
		}

		if err == io.EOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("failed to read audio, %v", err)
		}
	}
}
