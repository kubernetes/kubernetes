package session

import (
	"os"

	"github.com/aws/aws-sdk-go/internal/sdktesting"
)

func initSessionTestEnv() (oldEnv func()) {
	oldEnv = sdktesting.StashEnv()
	os.Setenv("AWS_CONFIG_FILE", "file_not_exists")
	os.Setenv("AWS_SHARED_CREDENTIALS_FILE", "file_not_exists")

	return oldEnv
}
