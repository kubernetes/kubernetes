// +build !windows

package s3manager

func defaultDownloadBufferProvider() WriterReadFromProvider {
	return nil
}
