package testutils

import (
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/docker/docker/api/types/events"
	timetypes "github.com/docker/docker/api/types/time"
)

var (
	reTimestamp  = `(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{9}(:?(:?(:?-|\+)\d{2}:\d{2})|Z))`
	reEventType  = `(?P<eventType>\w+)`
	reAction     = `(?P<action>\w+)`
	reID         = `(?P<id>[^\s]+)`
	reAttributes = `(\s\((?P<attributes>[^\)]+)\))?`
	reString     = fmt.Sprintf(`\A%s\s%s\s%s\s%s%s\z`, reTimestamp, reEventType, reAction, reID, reAttributes)

	// eventCliRegexp is a regular expression that matches all possible event outputs in the cli
	eventCliRegexp = regexp.MustCompile(reString)
)

// ScanMap turns an event string like the default ones formatted in the cli output
// and turns it into map.
func ScanMap(text string) map[string]string {
	matches := eventCliRegexp.FindAllStringSubmatch(text, -1)
	md := map[string]string{}
	if len(matches) == 0 {
		return md
	}

	names := eventCliRegexp.SubexpNames()
	for i, n := range matches[0] {
		md[names[i]] = n
	}
	return md
}

// Scan turns an event string like the default ones formatted in the cli output
// and turns it into an event message.
func Scan(text string) (*events.Message, error) {
	md := ScanMap(text)
	if len(md) == 0 {
		return nil, fmt.Errorf("text is not an event: %s", text)
	}

	f, err := timetypes.GetTimestamp(md["timestamp"], time.Now())
	if err != nil {
		return nil, err
	}

	t, tn, err := timetypes.ParseTimestamps(f, -1)
	if err != nil {
		return nil, err
	}

	attrs := make(map[string]string)
	for _, a := range strings.SplitN(md["attributes"], ", ", -1) {
		kv := strings.SplitN(a, "=", 2)
		attrs[kv[0]] = kv[1]
	}

	tu := time.Unix(t, tn)
	return &events.Message{
		Time:     t,
		TimeNano: tu.UnixNano(),
		Type:     md["eventType"],
		Action:   md["action"],
		Actor: events.Actor{
			ID:         md["id"],
			Attributes: attrs,
		},
	}, nil
}
