package runconfig

import (
	"fmt"
	"strings"
)

func parseNetMode(netMode string) (NetworkMode, error) {
	parts := strings.Split(netMode, ":")
	switch mode := parts[0]; mode {
	case "default", "none":
	default:
		return "", fmt.Errorf("invalid --net: %s", netMode)
	}
	return NetworkMode(netMode), nil
}

func validateNetMode(vals *validateNM) error {
	return nil
}
