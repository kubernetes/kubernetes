// Copyright 2013 Ooyala, Inc.

/*
Package statsd provides a Go dogstatsd client. Dogstatsd extends the popular statsd,
adding tags and histograms and pushing upstream to Datadog.

Refer to http://docs.datadoghq.com/guides/dogstatsd/ for information about DogStatsD.

Example Usage:

    // Create the client
    c, err := statsd.New("127.0.0.1:8125")
    if err != nil {
        log.Fatal(err)
    }
    // Prefix every metric with the app name
    c.Namespace = "flubber."
    // Send the EC2 availability zone as a tag with every metric
    c.Tags = append(c.Tags, "us-east-1a")
    err = c.Gauge("request.duration", 1.2, nil, 1)

statsd is based on go-statsd-client.
*/
package statsd

import (
	"bytes"
	"errors"
	"fmt"
	"math/rand"
	"net"
	"strconv"
	"strings"
	"sync"
	"time"
)

/*
OptimalPayloadSize defines the optimal payload size for a UDP datagram, 1432 bytes
is optimal for regular networks with an MTU of 1500 so datagrams don't get
fragmented. It's generally recommended not to fragment UDP datagrams as losing
a single fragment will cause the entire datagram to be lost.

This can be increased if your network has a greater MTU or you don't mind UDP
datagrams getting fragmented. The practical limit is MaxUDPPayloadSize
*/
const OptimalPayloadSize = 1432

/*
MaxUDPPayloadSize defines the maximum payload size for a UDP datagram.
Its value comes from the calculation: 65535 bytes Max UDP datagram size -
8byte UDP header - 60byte max IP headers
any number greater than that will see frames being cut out.
*/
const MaxUDPPayloadSize = 65467

// A Client is a handle for sending udp messages to dogstatsd.  It is safe to
// use one Client from multiple goroutines simultaneously.
type Client struct {
	conn net.Conn
	// Namespace to prepend to all statsd calls
	Namespace string
	// Tags are global tags to be added to every statsd call
	Tags []string
	// BufferLength is the length of the buffer in commands.
	bufferLength int
	flushTime    time.Duration
	commands     []string
	buffer       bytes.Buffer
	stop         bool
	sync.Mutex
}

// New returns a pointer to a new Client given an addr in the format "hostname:port".
func New(addr string) (*Client, error) {
	udpAddr, err := net.ResolveUDPAddr("udp", addr)
	if err != nil {
		return nil, err
	}
	conn, err := net.DialUDP("udp", nil, udpAddr)
	if err != nil {
		return nil, err
	}
	client := &Client{conn: conn}
	return client, nil
}

// NewBuffered returns a Client that buffers its output and sends it in chunks.
// Buflen is the length of the buffer in number of commands.
func NewBuffered(addr string, buflen int) (*Client, error) {
	client, err := New(addr)
	if err != nil {
		return nil, err
	}
	client.bufferLength = buflen
	client.commands = make([]string, 0, buflen)
	client.flushTime = time.Millisecond * 100
	go client.watch()
	return client, nil
}

// format a message from its name, value, tags and rate.  Also adds global
// namespace and tags.
func (c *Client) format(name, value string, tags []string, rate float64) string {
	var buf bytes.Buffer
	if c.Namespace != "" {
		buf.WriteString(c.Namespace)
	}
	buf.WriteString(name)
	buf.WriteString(":")
	buf.WriteString(value)
	if rate < 1 {
		buf.WriteString(`|@`)
		buf.WriteString(strconv.FormatFloat(rate, 'f', -1, 64))
	}

	// do not append to c.Tags directly, because it's shared
	// across all invocations of this function
	tagCopy := make([]string, len(c.Tags), len(c.Tags)+len(tags))
	copy(tagCopy, c.Tags)
	tags = append(tagCopy, tags...)
	if len(tags) > 0 {
		buf.WriteString("|#")
		buf.WriteString(tags[0])
		for _, tag := range tags[1:] {
			buf.WriteString(",")
			buf.WriteString(tag)
		}
	}
	return buf.String()
}

func (c *Client) watch() {
	for _ = range time.Tick(c.flushTime) {
		if c.stop {
			return
		}
		c.Lock()
		if len(c.commands) > 0 {
			// FIXME: eating error here
			c.flush()
		}
		c.Unlock()
	}
}

func (c *Client) append(cmd string) error {
	c.commands = append(c.commands, cmd)
	// if we should flush, lets do it
	if len(c.commands) == c.bufferLength {
		if err := c.flush(); err != nil {
			return err
		}
	}
	return nil
}

func (c *Client) joinMaxSize(cmds []string, sep string, maxSize int) ([][]byte, []int) {
	c.buffer.Reset() //clear buffer

	var frames [][]byte
	var ncmds []int
	sepBytes := []byte(sep)
	sepLen := len(sep)

	elem := 0
	for _, cmd := range cmds {
		needed := len(cmd)

		if elem != 0 {
			needed = needed + sepLen
		}

		if c.buffer.Len()+needed <= maxSize {
			if elem != 0 {
				c.buffer.Write(sepBytes)
			}
			c.buffer.WriteString(cmd)
			elem++
		} else {
			frames = append(frames, copyAndResetBuffer(&c.buffer))
			ncmds = append(ncmds, elem)
			// if cmd is bigger than maxSize it will get flushed on next loop
			c.buffer.WriteString(cmd)
			elem = 1
		}
	}

	//add whatever is left! if there's actually something
	if c.buffer.Len() > 0 {
		frames = append(frames, copyAndResetBuffer(&c.buffer))
		ncmds = append(ncmds, elem)
	}

	return frames, ncmds
}

func copyAndResetBuffer(buf *bytes.Buffer) []byte {
	tmpBuf := make([]byte, buf.Len())
	copy(tmpBuf, buf.Bytes())
	buf.Reset()
	return tmpBuf
}

// flush the commands in the buffer.  Lock must be held by caller.
func (c *Client) flush() error {
	frames, flushable := c.joinMaxSize(c.commands, "\n", OptimalPayloadSize)
	var err error
	cmdsFlushed := 0
	for i, data := range frames {
		_, e := c.conn.Write(data)
		if e != nil {
			err = e
			break
		}
		cmdsFlushed += flushable[i]
	}

	// clear the slice with a slice op, doesn't realloc
	if cmdsFlushed == len(c.commands) {
		c.commands = c.commands[:0]
	} else {
		//this case will cause a future realloc...
		// drop problematic command though (sorry).
		c.commands = c.commands[cmdsFlushed+1:]
	}
	return err
}

func (c *Client) sendMsg(msg string) error {
	// if this client is buffered, then we'll just append this
	c.Lock()
	defer c.Unlock()
	if c.bufferLength > 0 {
		// return an error if message is bigger than OptimalPayloadSize
		if len(msg) > MaxUDPPayloadSize {
			return errors.New("message size exceeds MaxUDPPayloadSize")
		}
		return c.append(msg)
	}
	_, err := c.conn.Write([]byte(msg))
	return err
}

// send handles sampling and sends the message over UDP. It also adds global namespace prefixes and tags.
func (c *Client) send(name, value string, tags []string, rate float64) error {
	if c == nil {
		return nil
	}
	if rate < 1 && rand.Float64() > rate {
		return nil
	}
	data := c.format(name, value, tags, rate)
	return c.sendMsg(data)
}

// Gauge measures the value of a metric at a particular time.
func (c *Client) Gauge(name string, value float64, tags []string, rate float64) error {
	stat := fmt.Sprintf("%f|g", value)
	return c.send(name, stat, tags, rate)
}

// Count tracks how many times something happened per second.
func (c *Client) Count(name string, value int64, tags []string, rate float64) error {
	stat := fmt.Sprintf("%d|c", value)
	return c.send(name, stat, tags, rate)
}

// Histogram tracks the statistical distribution of a set of values.
func (c *Client) Histogram(name string, value float64, tags []string, rate float64) error {
	stat := fmt.Sprintf("%f|h", value)
	return c.send(name, stat, tags, rate)
}

// Decr is just Count of 1
func (c *Client) Decr(name string, tags []string, rate float64) error {
	return c.send(name, "-1|c", tags, rate)
}

// Incr is just Count of 1
func (c *Client) Incr(name string, tags []string, rate float64) error {
	return c.send(name, "1|c", tags, rate)
}

// Set counts the number of unique elements in a group.
func (c *Client) Set(name string, value string, tags []string, rate float64) error {
	stat := fmt.Sprintf("%s|s", value)
	return c.send(name, stat, tags, rate)
}

// Timing sends timing information, it is an alias for TimeInMilliseconds
func (c *Client) Timing(name string, value time.Duration, tags []string, rate float64) error {
	return c.TimeInMilliseconds(name, value.Seconds()*1000, tags, rate)
}

// TimeInMilliseconds sends timing information in milliseconds.
// It is flushed by statsd with percentiles, mean and other info (https://github.com/etsy/statsd/blob/master/docs/metric_types.md#timing)
func (c *Client) TimeInMilliseconds(name string, value float64, tags []string, rate float64) error {
	stat := fmt.Sprintf("%f|ms", value)
	return c.send(name, stat, tags, rate)
}

// Event sends the provided Event.
func (c *Client) Event(e *Event) error {
	stat, err := e.Encode(c.Tags...)
	if err != nil {
		return err
	}
	return c.sendMsg(stat)
}

// SimpleEvent sends an event with the provided title and text.
func (c *Client) SimpleEvent(title, text string) error {
	e := NewEvent(title, text)
	return c.Event(e)
}

// ServiceCheck sends the provided ServiceCheck.
func (c *Client) ServiceCheck(sc *ServiceCheck) error {
	stat, err := sc.Encode(c.Tags...)
	if err != nil {
		return err
	}
	return c.sendMsg(stat)
}

// SimpleServiceCheck sends an serviceCheck with the provided name and status.
func (c *Client) SimpleServiceCheck(name string, status serviceCheckStatus) error {
	sc := NewServiceCheck(name, status)
	return c.ServiceCheck(sc)
}

// Close the client connection.
func (c *Client) Close() error {
	if c == nil {
		return nil
	}
	c.stop = true
	return c.conn.Close()
}

// Events support

type eventAlertType string

const (
	// Info is the "info" AlertType for events
	Info eventAlertType = "info"
	// Error is the "error" AlertType for events
	Error eventAlertType = "error"
	// Warning is the "warning" AlertType for events
	Warning eventAlertType = "warning"
	// Success is the "success" AlertType for events
	Success eventAlertType = "success"
)

type eventPriority string

const (
	// Normal is the "normal" Priority for events
	Normal eventPriority = "normal"
	// Low is the "low" Priority for events
	Low eventPriority = "low"
)

// An Event is an object that can be posted to your DataDog event stream.
type Event struct {
	// Title of the event.  Required.
	Title string
	// Text is the description of the event.  Required.
	Text string
	// Timestamp is a timestamp for the event.  If not provided, the dogstatsd
	// server will set this to the current time.
	Timestamp time.Time
	// Hostname for the event.
	Hostname string
	// AggregationKey groups this event with others of the same key.
	AggregationKey string
	// Priority of the event.  Can be statsd.Low or statsd.Normal.
	Priority eventPriority
	// SourceTypeName is a source type for the event.
	SourceTypeName string
	// AlertType can be statsd.Info, statsd.Error, statsd.Warning, or statsd.Success.
	// If absent, the default value applied by the dogstatsd server is Info.
	AlertType eventAlertType
	// Tags for the event.
	Tags []string
}

// NewEvent creates a new event with the given title and text.  Error checking
// against these values is done at send-time, or upon running e.Check.
func NewEvent(title, text string) *Event {
	return &Event{
		Title: title,
		Text:  text,
	}
}

// Check verifies that an event is valid.
func (e Event) Check() error {
	if len(e.Title) == 0 {
		return fmt.Errorf("statsd.Event title is required")
	}
	if len(e.Text) == 0 {
		return fmt.Errorf("statsd.Event text is required")
	}
	return nil
}

// Encode returns the dogstatsd wire protocol representation for an event.
// Tags may be passed which will be added to the encoded output but not to
// the Event's list of tags, eg. for default tags.
func (e Event) Encode(tags ...string) (string, error) {
	err := e.Check()
	if err != nil {
		return "", err
	}
	text := e.escapedText()

	var buffer bytes.Buffer
	buffer.WriteString("_e{")
	buffer.WriteString(strconv.FormatInt(int64(len(e.Title)), 10))
	buffer.WriteRune(',')
	buffer.WriteString(strconv.FormatInt(int64(len(text)), 10))
	buffer.WriteString("}:")
	buffer.WriteString(e.Title)
	buffer.WriteRune('|')
	buffer.WriteString(text)

	if !e.Timestamp.IsZero() {
		buffer.WriteString("|d:")
		buffer.WriteString(strconv.FormatInt(int64(e.Timestamp.Unix()), 10))
	}

	if len(e.Hostname) != 0 {
		buffer.WriteString("|h:")
		buffer.WriteString(e.Hostname)
	}

	if len(e.AggregationKey) != 0 {
		buffer.WriteString("|k:")
		buffer.WriteString(e.AggregationKey)

	}

	if len(e.Priority) != 0 {
		buffer.WriteString("|p:")
		buffer.WriteString(string(e.Priority))
	}

	if len(e.SourceTypeName) != 0 {
		buffer.WriteString("|s:")
		buffer.WriteString(e.SourceTypeName)
	}

	if len(e.AlertType) != 0 {
		buffer.WriteString("|t:")
		buffer.WriteString(string(e.AlertType))
	}

	if len(tags)+len(e.Tags) > 0 {
		all := make([]string, 0, len(tags)+len(e.Tags))
		all = append(all, tags...)
		all = append(all, e.Tags...)
		buffer.WriteString("|#")
		buffer.WriteString(all[0])
		for _, tag := range all[1:] {
			buffer.WriteString(",")
			buffer.WriteString(tag)
		}
	}

	return buffer.String(), nil
}

// ServiceCheck support

type serviceCheckStatus byte

const (
	// Ok is the "ok" ServiceCheck status
	Ok serviceCheckStatus = 0
	// Warn is the "warning" ServiceCheck status
	Warn serviceCheckStatus = 1
	// Critical is the "critical" ServiceCheck status
	Critical serviceCheckStatus = 2
	// Unknown is the "unknown" ServiceCheck status
	Unknown serviceCheckStatus = 3
)

// An ServiceCheck is an object that contains status of DataDog service check.
type ServiceCheck struct {
	// Name of the service check.  Required.
	Name string
	// Status of service check.  Required.
	Status serviceCheckStatus
	// Timestamp is a timestamp for the serviceCheck.  If not provided, the dogstatsd
	// server will set this to the current time.
	Timestamp time.Time
	// Hostname for the serviceCheck.
	Hostname string
	// A message describing the current state of the serviceCheck.
	Message string
	// Tags for the serviceCheck.
	Tags []string
}

// NewServiceCheck creates a new serviceCheck with the given name and status.  Error checking
// against these values is done at send-time, or upon running sc.Check.
func NewServiceCheck(name string, status serviceCheckStatus) *ServiceCheck {
	return &ServiceCheck{
		Name:   name,
		Status: status,
	}
}

// Check verifies that an event is valid.
func (sc ServiceCheck) Check() error {
	if len(sc.Name) == 0 {
		return fmt.Errorf("statsd.ServiceCheck name is required")
	}
	if byte(sc.Status) < 0 || byte(sc.Status) > 3 {
		return fmt.Errorf("statsd.ServiceCheck status has invalid value")
	}
	return nil
}

// Encode returns the dogstatsd wire protocol representation for an serviceCheck.
// Tags may be passed which will be added to the encoded output but not to
// the Event's list of tags, eg. for default tags.
func (sc ServiceCheck) Encode(tags ...string) (string, error) {
	err := sc.Check()
	if err != nil {
		return "", err
	}
	message := sc.escapedMessage()

	var buffer bytes.Buffer
	buffer.WriteString("_sc|")
	buffer.WriteString(sc.Name)
	buffer.WriteRune('|')
	buffer.WriteString(strconv.FormatInt(int64(sc.Status), 10))

	if !sc.Timestamp.IsZero() {
		buffer.WriteString("|d:")
		buffer.WriteString(strconv.FormatInt(int64(sc.Timestamp.Unix()), 10))
	}

	if len(sc.Hostname) != 0 {
		buffer.WriteString("|h:")
		buffer.WriteString(sc.Hostname)
	}

	if len(tags)+len(sc.Tags) > 0 {
		all := make([]string, 0, len(tags)+len(sc.Tags))
		all = append(all, tags...)
		all = append(all, sc.Tags...)
		buffer.WriteString("|#")
		buffer.WriteString(all[0])
		for _, tag := range all[1:] {
			buffer.WriteString(",")
			buffer.WriteString(tag)
		}
	}

	if len(message) != 0 {
		buffer.WriteString("|m:")
		buffer.WriteString(message)
	}

	return buffer.String(), nil
}

func (e Event) escapedText() string {
	return strings.Replace(e.Text, "\n", "\\n", -1)
}

func (sc ServiceCheck) escapedMessage() string {
	msg := strings.Replace(sc.Message, "\n", "\\n", -1)
	return strings.Replace(msg, "m:", `m\:`, -1)
}
