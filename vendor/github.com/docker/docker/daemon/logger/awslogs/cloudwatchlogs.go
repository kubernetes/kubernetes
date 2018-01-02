// Package awslogs provides the logdriver for forwarding container logs to Amazon CloudWatch Logs
package awslogs

import (
	"bytes"
	"fmt"
	"os"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/ec2metadata"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/cloudwatchlogs"
	"github.com/docker/docker/daemon/logger"
	"github.com/docker/docker/daemon/logger/loggerutils"
	"github.com/docker/docker/dockerversion"
	"github.com/docker/docker/pkg/templates"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

const (
	name                  = "awslogs"
	regionKey             = "awslogs-region"
	regionEnvKey          = "AWS_REGION"
	logGroupKey           = "awslogs-group"
	logStreamKey          = "awslogs-stream"
	logCreateGroupKey     = "awslogs-create-group"
	tagKey                = "tag"
	datetimeFormatKey     = "awslogs-datetime-format"
	multilinePatternKey   = "awslogs-multiline-pattern"
	batchPublishFrequency = 5 * time.Second

	// See: http://docs.aws.amazon.com/AmazonCloudWatchLogs/latest/APIReference/API_PutLogEvents.html
	perEventBytes          = 26
	maximumBytesPerPut     = 1048576
	maximumLogEventsPerPut = 10000

	// See: http://docs.aws.amazon.com/AmazonCloudWatch/latest/DeveloperGuide/cloudwatch_limits.html
	maximumBytesPerEvent = 262144 - perEventBytes

	resourceAlreadyExistsCode = "ResourceAlreadyExistsException"
	dataAlreadyAcceptedCode   = "DataAlreadyAcceptedException"
	invalidSequenceTokenCode  = "InvalidSequenceTokenException"
	resourceNotFoundCode      = "ResourceNotFoundException"

	userAgentHeader = "User-Agent"
)

type logStream struct {
	logStreamName    string
	logGroupName     string
	logCreateGroup   bool
	multilinePattern *regexp.Regexp
	client           api
	messages         chan *logger.Message
	lock             sync.RWMutex
	closed           bool
	sequenceToken    *string
}

type api interface {
	CreateLogGroup(*cloudwatchlogs.CreateLogGroupInput) (*cloudwatchlogs.CreateLogGroupOutput, error)
	CreateLogStream(*cloudwatchlogs.CreateLogStreamInput) (*cloudwatchlogs.CreateLogStreamOutput, error)
	PutLogEvents(*cloudwatchlogs.PutLogEventsInput) (*cloudwatchlogs.PutLogEventsOutput, error)
}

type regionFinder interface {
	Region() (string, error)
}

type wrappedEvent struct {
	inputLogEvent *cloudwatchlogs.InputLogEvent
	insertOrder   int
}
type byTimestamp []wrappedEvent

// init registers the awslogs driver
func init() {
	if err := logger.RegisterLogDriver(name, New); err != nil {
		logrus.Fatal(err)
	}
	if err := logger.RegisterLogOptValidator(name, ValidateLogOpt); err != nil {
		logrus.Fatal(err)
	}
}

// New creates an awslogs logger using the configuration passed in on the
// context.  Supported context configuration variables are awslogs-region,
// awslogs-group, awslogs-stream, awslogs-create-group, awslogs-multiline-pattern
// and awslogs-datetime-format.  When available, configuration is
// also taken from environment variables AWS_REGION, AWS_ACCESS_KEY_ID,
// AWS_SECRET_ACCESS_KEY, the shared credentials file (~/.aws/credentials), and
// the EC2 Instance Metadata Service.
func New(info logger.Info) (logger.Logger, error) {
	logGroupName := info.Config[logGroupKey]
	logStreamName, err := loggerutils.ParseLogTag(info, "{{.FullID}}")
	if err != nil {
		return nil, err
	}
	logCreateGroup := false
	if info.Config[logCreateGroupKey] != "" {
		logCreateGroup, err = strconv.ParseBool(info.Config[logCreateGroupKey])
		if err != nil {
			return nil, err
		}
	}

	if info.Config[logStreamKey] != "" {
		logStreamName = info.Config[logStreamKey]
	}

	multilinePattern, err := parseMultilineOptions(info)
	if err != nil {
		return nil, err
	}

	client, err := newAWSLogsClient(info)
	if err != nil {
		return nil, err
	}
	containerStream := &logStream{
		logStreamName:    logStreamName,
		logGroupName:     logGroupName,
		logCreateGroup:   logCreateGroup,
		multilinePattern: multilinePattern,
		client:           client,
		messages:         make(chan *logger.Message, 4096),
	}
	err = containerStream.create()
	if err != nil {
		return nil, err
	}
	go containerStream.collectBatch()

	return containerStream, nil
}

// Parses awslogs-multiline-pattern and awslogs-datetime-format options
// If awslogs-datetime-format is present, convert the format from strftime
// to regexp and return.
// If awslogs-multiline-pattern is present, compile regexp and return
func parseMultilineOptions(info logger.Info) (*regexp.Regexp, error) {
	dateTimeFormat := info.Config[datetimeFormatKey]
	multilinePatternKey := info.Config[multilinePatternKey]
	// strftime input is parsed into a regular expression
	if dateTimeFormat != "" {
		// %. matches each strftime format sequence and ReplaceAllStringFunc
		// looks up each format sequence in the conversion table strftimeToRegex
		// to replace with a defined regular expression
		r := regexp.MustCompile("%.")
		multilinePatternKey = r.ReplaceAllStringFunc(dateTimeFormat, func(s string) string {
			return strftimeToRegex[s]
		})
	}
	if multilinePatternKey != "" {
		multilinePattern, err := regexp.Compile(multilinePatternKey)
		if err != nil {
			return nil, errors.Wrapf(err, "awslogs could not parse multiline pattern key %q", multilinePatternKey)
		}
		return multilinePattern, nil
	}
	return nil, nil
}

// Maps strftime format strings to regex
var strftimeToRegex = map[string]string{
	/*weekdayShort          */ `%a`: `(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)`,
	/*weekdayFull           */ `%A`: `(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)`,
	/*weekdayZeroIndex      */ `%w`: `[0-6]`,
	/*dayZeroPadded         */ `%d`: `(?:0[1-9]|[1,2][0-9]|3[0,1])`,
	/*monthShort            */ `%b`: `(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)`,
	/*monthFull             */ `%B`: `(?:January|February|March|April|May|June|July|August|September|October|November|December)`,
	/*monthZeroPadded       */ `%m`: `(?:0[1-9]|1[0-2])`,
	/*yearCentury           */ `%Y`: `\d{4}`,
	/*yearZeroPadded        */ `%y`: `\d{2}`,
	/*hour24ZeroPadded      */ `%H`: `(?:[0,1][0-9]|2[0-3])`,
	/*hour12ZeroPadded      */ `%I`: `(?:0[0-9]|1[0-2])`,
	/*AM or PM              */ `%p`: "[A,P]M",
	/*minuteZeroPadded      */ `%M`: `[0-5][0-9]`,
	/*secondZeroPadded      */ `%S`: `[0-5][0-9]`,
	/*microsecondZeroPadded */ `%f`: `\d{6}`,
	/*utcOffset             */ `%z`: `[+-]\d{4}`,
	/*tzName                */ `%Z`: `[A-Z]{1,4}T`,
	/*dayOfYearZeroPadded   */ `%j`: `(?:0[0-9][1-9]|[1,2][0-9][0-9]|3[0-5][0-9]|36[0-6])`,
	/*milliseconds          */ `%L`: `\.\d{3}`,
}

func parseLogGroup(info logger.Info, groupTemplate string) (string, error) {
	tmpl, err := templates.NewParse("log-group", groupTemplate)
	if err != nil {
		return "", err
	}
	buf := new(bytes.Buffer)
	if err := tmpl.Execute(buf, &info); err != nil {
		return "", err
	}

	return buf.String(), nil
}

// newRegionFinder is a variable such that the implementation
// can be swapped out for unit tests.
var newRegionFinder = func() regionFinder {
	return ec2metadata.New(session.New())
}

// newAWSLogsClient creates the service client for Amazon CloudWatch Logs.
// Customizations to the default client from the SDK include a Docker-specific
// User-Agent string and automatic region detection using the EC2 Instance
// Metadata Service when region is otherwise unspecified.
func newAWSLogsClient(info logger.Info) (api, error) {
	var region *string
	if os.Getenv(regionEnvKey) != "" {
		region = aws.String(os.Getenv(regionEnvKey))
	}
	if info.Config[regionKey] != "" {
		region = aws.String(info.Config[regionKey])
	}
	if region == nil || *region == "" {
		logrus.Info("Trying to get region from EC2 Metadata")
		ec2MetadataClient := newRegionFinder()
		r, err := ec2MetadataClient.Region()
		if err != nil {
			logrus.WithFields(logrus.Fields{
				"error": err,
			}).Error("Could not get region from EC2 metadata, environment, or log option")
			return nil, errors.New("Cannot determine region for awslogs driver")
		}
		region = &r
	}
	logrus.WithFields(logrus.Fields{
		"region": *region,
	}).Debug("Created awslogs client")

	client := cloudwatchlogs.New(session.New(), aws.NewConfig().WithRegion(*region))

	client.Handlers.Build.PushBackNamed(request.NamedHandler{
		Name: "DockerUserAgentHandler",
		Fn: func(r *request.Request) {
			currentAgent := r.HTTPRequest.Header.Get(userAgentHeader)
			r.HTTPRequest.Header.Set(userAgentHeader,
				fmt.Sprintf("Docker %s (%s) %s",
					dockerversion.Version, runtime.GOOS, currentAgent))
		},
	})
	return client, nil
}

// Name returns the name of the awslogs logging driver
func (l *logStream) Name() string {
	return name
}

// Log submits messages for logging by an instance of the awslogs logging driver
func (l *logStream) Log(msg *logger.Message) error {
	l.lock.RLock()
	defer l.lock.RUnlock()
	if !l.closed {
		l.messages <- msg
	}
	return nil
}

// Close closes the instance of the awslogs logging driver
func (l *logStream) Close() error {
	l.lock.Lock()
	defer l.lock.Unlock()
	if !l.closed {
		close(l.messages)
	}
	l.closed = true
	return nil
}

// create creates log group and log stream for the instance of the awslogs logging driver
func (l *logStream) create() error {
	if err := l.createLogStream(); err != nil {
		if l.logCreateGroup {
			if awsErr, ok := err.(awserr.Error); ok && awsErr.Code() == resourceNotFoundCode {
				if err := l.createLogGroup(); err != nil {
					return err
				}
				return l.createLogStream()
			}
		}
		return err
	}

	return nil
}

// createLogGroup creates a log group for the instance of the awslogs logging driver
func (l *logStream) createLogGroup() error {
	if _, err := l.client.CreateLogGroup(&cloudwatchlogs.CreateLogGroupInput{
		LogGroupName: aws.String(l.logGroupName),
	}); err != nil {
		if awsErr, ok := err.(awserr.Error); ok {
			fields := logrus.Fields{
				"errorCode":      awsErr.Code(),
				"message":        awsErr.Message(),
				"origError":      awsErr.OrigErr(),
				"logGroupName":   l.logGroupName,
				"logCreateGroup": l.logCreateGroup,
			}
			if awsErr.Code() == resourceAlreadyExistsCode {
				// Allow creation to succeed
				logrus.WithFields(fields).Info("Log group already exists")
				return nil
			}
			logrus.WithFields(fields).Error("Failed to create log group")
		}
		return err
	}
	return nil
}

// createLogStream creates a log stream for the instance of the awslogs logging driver
func (l *logStream) createLogStream() error {
	input := &cloudwatchlogs.CreateLogStreamInput{
		LogGroupName:  aws.String(l.logGroupName),
		LogStreamName: aws.String(l.logStreamName),
	}

	_, err := l.client.CreateLogStream(input)

	if err != nil {
		if awsErr, ok := err.(awserr.Error); ok {
			fields := logrus.Fields{
				"errorCode":     awsErr.Code(),
				"message":       awsErr.Message(),
				"origError":     awsErr.OrigErr(),
				"logGroupName":  l.logGroupName,
				"logStreamName": l.logStreamName,
			}
			if awsErr.Code() == resourceAlreadyExistsCode {
				// Allow creation to succeed
				logrus.WithFields(fields).Info("Log stream already exists")
				return nil
			}
			logrus.WithFields(fields).Error("Failed to create log stream")
		}
	}
	return err
}

// newTicker is used for time-based batching.  newTicker is a variable such
// that the implementation can be swapped out for unit tests.
var newTicker = func(freq time.Duration) *time.Ticker {
	return time.NewTicker(freq)
}

// collectBatch executes as a goroutine to perform batching of log events for
// submission to the log stream.  If the awslogs-multiline-pattern or
// awslogs-datetime-format options have been configured, multiline processing
// is enabled, where log messages are stored in an event buffer until a multiline
// pattern match is found, at which point the messages in the event buffer are
// pushed to CloudWatch logs as a single log event.  Multiline messages are processed
// according to the maximumBytesPerPut constraint, and the implementation only
// allows for messages to be buffered for a maximum of 2*batchPublishFrequency
// seconds.  When events are ready to be processed for submission to CloudWatch
// Logs, the processEvents method is called.  If a multiline pattern is not
// configured, log events are submitted to the processEvents method immediately.
func (l *logStream) collectBatch() {
	timer := newTicker(batchPublishFrequency)
	var events []wrappedEvent
	var eventBuffer []byte
	var eventBufferTimestamp int64
	for {
		select {
		case t := <-timer.C:
			// If event buffer is older than batch publish frequency flush the event buffer
			if eventBufferTimestamp > 0 && len(eventBuffer) > 0 {
				eventBufferAge := t.UnixNano()/int64(time.Millisecond) - eventBufferTimestamp
				eventBufferExpired := eventBufferAge > int64(batchPublishFrequency)/int64(time.Millisecond)
				eventBufferNegative := eventBufferAge < 0
				if eventBufferExpired || eventBufferNegative {
					events = l.processEvent(events, eventBuffer, eventBufferTimestamp)
					eventBuffer = eventBuffer[:0]
				}
			}
			l.publishBatch(events)
			events = events[:0]
		case msg, more := <-l.messages:
			if !more {
				// Flush event buffer and release resources
				events = l.processEvent(events, eventBuffer, eventBufferTimestamp)
				eventBuffer = eventBuffer[:0]
				l.publishBatch(events)
				events = events[:0]
				return
			}
			if eventBufferTimestamp == 0 {
				eventBufferTimestamp = msg.Timestamp.UnixNano() / int64(time.Millisecond)
			}
			unprocessedLine := msg.Line
			if l.multilinePattern != nil {
				if l.multilinePattern.Match(unprocessedLine) || len(eventBuffer)+len(unprocessedLine) > maximumBytesPerEvent {
					// This is a new log event or we will exceed max bytes per event
					// so flush the current eventBuffer to events and reset timestamp
					events = l.processEvent(events, eventBuffer, eventBufferTimestamp)
					eventBufferTimestamp = msg.Timestamp.UnixNano() / int64(time.Millisecond)
					eventBuffer = eventBuffer[:0]
				}
				// Append new line
				processedLine := append(unprocessedLine, "\n"...)
				eventBuffer = append(eventBuffer, processedLine...)
				logger.PutMessage(msg)
			} else {
				events = l.processEvent(events, unprocessedLine, msg.Timestamp.UnixNano()/int64(time.Millisecond))
				logger.PutMessage(msg)
			}
		}
	}
}

// processEvent processes log events that are ready for submission to CloudWatch
// logs.  Batching is performed on time- and size-bases.  Time-based batching
// occurs at a 5 second interval (defined in the batchPublishFrequency const).
// Size-based batching is performed on the maximum number of events per batch
// (defined in maximumLogEventsPerPut) and the maximum number of total bytes in a
// batch (defined in maximumBytesPerPut).  Log messages are split by the maximum
// bytes per event (defined in maximumBytesPerEvent).  There is a fixed per-event
// byte overhead (defined in perEventBytes) which is accounted for in split- and
// batch-calculations.
func (l *logStream) processEvent(events []wrappedEvent, unprocessedLine []byte, timestamp int64) []wrappedEvent {
	bytes := 0
	for len(unprocessedLine) > 0 {
		// Split line length so it does not exceed the maximum
		lineBytes := len(unprocessedLine)
		if lineBytes > maximumBytesPerEvent {
			lineBytes = maximumBytesPerEvent
		}
		line := unprocessedLine[:lineBytes]
		unprocessedLine = unprocessedLine[lineBytes:]
		if (len(events) >= maximumLogEventsPerPut) || (bytes+lineBytes+perEventBytes > maximumBytesPerPut) {
			// Publish an existing batch if it's already over the maximum number of events or if adding this
			// event would push it over the maximum number of total bytes.
			l.publishBatch(events)
			events = events[:0]
			bytes = 0
		}
		events = append(events, wrappedEvent{
			inputLogEvent: &cloudwatchlogs.InputLogEvent{
				Message:   aws.String(string(line)),
				Timestamp: aws.Int64(timestamp),
			},
			insertOrder: len(events),
		})
		bytes += (lineBytes + perEventBytes)
	}
	return events
}

// publishBatch calls PutLogEvents for a given set of InputLogEvents,
// accounting for sequencing requirements (each request must reference the
// sequence token returned by the previous request).
func (l *logStream) publishBatch(events []wrappedEvent) {
	if len(events) == 0 {
		return
	}

	// events in a batch must be sorted by timestamp
	// see http://docs.aws.amazon.com/AmazonCloudWatchLogs/latest/APIReference/API_PutLogEvents.html
	sort.Sort(byTimestamp(events))
	cwEvents := unwrapEvents(events)

	nextSequenceToken, err := l.putLogEvents(cwEvents, l.sequenceToken)

	if err != nil {
		if awsErr, ok := err.(awserr.Error); ok {
			if awsErr.Code() == dataAlreadyAcceptedCode {
				// already submitted, just grab the correct sequence token
				parts := strings.Split(awsErr.Message(), " ")
				nextSequenceToken = &parts[len(parts)-1]
				logrus.WithFields(logrus.Fields{
					"errorCode":     awsErr.Code(),
					"message":       awsErr.Message(),
					"logGroupName":  l.logGroupName,
					"logStreamName": l.logStreamName,
				}).Info("Data already accepted, ignoring error")
				err = nil
			} else if awsErr.Code() == invalidSequenceTokenCode {
				// sequence code is bad, grab the correct one and retry
				parts := strings.Split(awsErr.Message(), " ")
				token := parts[len(parts)-1]
				nextSequenceToken, err = l.putLogEvents(cwEvents, &token)
			}
		}
	}
	if err != nil {
		logrus.Error(err)
	} else {
		l.sequenceToken = nextSequenceToken
	}
}

// putLogEvents wraps the PutLogEvents API
func (l *logStream) putLogEvents(events []*cloudwatchlogs.InputLogEvent, sequenceToken *string) (*string, error) {
	input := &cloudwatchlogs.PutLogEventsInput{
		LogEvents:     events,
		SequenceToken: sequenceToken,
		LogGroupName:  aws.String(l.logGroupName),
		LogStreamName: aws.String(l.logStreamName),
	}
	resp, err := l.client.PutLogEvents(input)
	if err != nil {
		if awsErr, ok := err.(awserr.Error); ok {
			logrus.WithFields(logrus.Fields{
				"errorCode":     awsErr.Code(),
				"message":       awsErr.Message(),
				"origError":     awsErr.OrigErr(),
				"logGroupName":  l.logGroupName,
				"logStreamName": l.logStreamName,
			}).Error("Failed to put log events")
		}
		return nil, err
	}
	return resp.NextSequenceToken, nil
}

// ValidateLogOpt looks for awslogs-specific log options awslogs-region,
// awslogs-group, awslogs-stream, awslogs-create-group, awslogs-datetime-format,
// awslogs-multiline-pattern
func ValidateLogOpt(cfg map[string]string) error {
	for key := range cfg {
		switch key {
		case logGroupKey:
		case logStreamKey:
		case logCreateGroupKey:
		case regionKey:
		case tagKey:
		case datetimeFormatKey:
		case multilinePatternKey:
		default:
			return fmt.Errorf("unknown log opt '%s' for %s log driver", key, name)
		}
	}
	if cfg[logGroupKey] == "" {
		return fmt.Errorf("must specify a value for log opt '%s'", logGroupKey)
	}
	if cfg[logCreateGroupKey] != "" {
		if _, err := strconv.ParseBool(cfg[logCreateGroupKey]); err != nil {
			return fmt.Errorf("must specify valid value for log opt '%s': %v", logCreateGroupKey, err)
		}
	}
	_, datetimeFormatKeyExists := cfg[datetimeFormatKey]
	_, multilinePatternKeyExists := cfg[multilinePatternKey]
	if datetimeFormatKeyExists && multilinePatternKeyExists {
		return fmt.Errorf("you cannot configure log opt '%s' and '%s' at the same time", datetimeFormatKey, multilinePatternKey)
	}
	return nil
}

// Len returns the length of a byTimestamp slice.  Len is required by the
// sort.Interface interface.
func (slice byTimestamp) Len() int {
	return len(slice)
}

// Less compares two values in a byTimestamp slice by Timestamp.  Less is
// required by the sort.Interface interface.
func (slice byTimestamp) Less(i, j int) bool {
	iTimestamp, jTimestamp := int64(0), int64(0)
	if slice != nil && slice[i].inputLogEvent.Timestamp != nil {
		iTimestamp = *slice[i].inputLogEvent.Timestamp
	}
	if slice != nil && slice[j].inputLogEvent.Timestamp != nil {
		jTimestamp = *slice[j].inputLogEvent.Timestamp
	}
	if iTimestamp == jTimestamp {
		return slice[i].insertOrder < slice[j].insertOrder
	}
	return iTimestamp < jTimestamp
}

// Swap swaps two values in a byTimestamp slice with each other.  Swap is
// required by the sort.Interface interface.
func (slice byTimestamp) Swap(i, j int) {
	slice[i], slice[j] = slice[j], slice[i]
}

func unwrapEvents(events []wrappedEvent) []*cloudwatchlogs.InputLogEvent {
	cwEvents := make([]*cloudwatchlogs.InputLogEvent, len(events))
	for i, input := range events {
		cwEvents[i] = input.inputLogEvent
	}
	return cwEvents
}
