package csm

import (
	"encoding/json"
	"net"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
)

// Reporter will gather metrics of API requests made and
// send those metrics to the CSM endpoint.
type Reporter struct {
	clientID  string
	url       string
	conn      net.Conn
	metricsCh metricChan
	done      chan struct{}
}

var (
	sender *Reporter
)

func connect(url string) error {
	const network = "udp"
	if err := sender.connect(network, url); err != nil {
		return err
	}

	if sender.done == nil {
		sender.done = make(chan struct{})
		go sender.start()
	}

	return nil
}

func newReporter(clientID, url string) *Reporter {
	return &Reporter{
		clientID:  clientID,
		url:       url,
		metricsCh: newMetricChan(MetricsChannelSize),
	}
}

func (rep *Reporter) sendAPICallAttemptMetric(r *request.Request) {
	if rep == nil {
		return
	}

	now := time.Now()
	creds, _ := r.Config.Credentials.Get()

	m := metric{
		ClientID:  aws.String(rep.clientID),
		API:       aws.String(r.Operation.Name),
		Service:   aws.String(r.ClientInfo.ServiceID),
		Timestamp: (*metricTime)(&now),
		UserAgent: aws.String(r.HTTPRequest.Header.Get("User-Agent")),
		Region:    r.Config.Region,
		Type:      aws.String("ApiCallAttempt"),
		Version:   aws.Int(1),

		XAmzRequestID: aws.String(r.RequestID),

		AttemptLatency: aws.Int(int(now.Sub(r.AttemptTime).Nanoseconds() / int64(time.Millisecond))),
		AccessKey:      aws.String(creds.AccessKeyID),
	}

	if r.HTTPResponse != nil {
		m.HTTPStatusCode = aws.Int(r.HTTPResponse.StatusCode)
	}

	if r.Error != nil {
		if awserr, ok := r.Error.(awserr.Error); ok {
			m.SetException(getMetricException(awserr))
		}
	}

	m.TruncateFields()
	rep.metricsCh.Push(m)
}

func getMetricException(err awserr.Error) metricException {
	msg := err.Error()
	code := err.Code()

	switch code {
	case request.ErrCodeRequestError,
		request.ErrCodeSerialization,
		request.CanceledErrorCode:
		return sdkException{
			requestException{exception: code, message: msg},
		}
	default:
		return awsException{
			requestException{exception: code, message: msg},
		}
	}
}

func (rep *Reporter) sendAPICallMetric(r *request.Request) {
	if rep == nil {
		return
	}

	now := time.Now()
	m := metric{
		ClientID:           aws.String(rep.clientID),
		API:                aws.String(r.Operation.Name),
		Service:            aws.String(r.ClientInfo.ServiceID),
		Timestamp:          (*metricTime)(&now),
		UserAgent:          aws.String(r.HTTPRequest.Header.Get("User-Agent")),
		Type:               aws.String("ApiCall"),
		AttemptCount:       aws.Int(r.RetryCount + 1),
		Region:             r.Config.Region,
		Latency:            aws.Int(int(time.Since(r.Time) / time.Millisecond)),
		XAmzRequestID:      aws.String(r.RequestID),
		MaxRetriesExceeded: aws.Int(boolIntValue(r.RetryCount >= r.MaxRetries())),
	}

	if r.HTTPResponse != nil {
		m.FinalHTTPStatusCode = aws.Int(r.HTTPResponse.StatusCode)
	}

	if r.Error != nil {
		if awserr, ok := r.Error.(awserr.Error); ok {
			m.SetFinalException(getMetricException(awserr))
		}
	}

	m.TruncateFields()

	// TODO: Probably want to figure something out for logging dropped
	// metrics
	rep.metricsCh.Push(m)
}

func (rep *Reporter) connect(network, url string) error {
	if rep.conn != nil {
		rep.conn.Close()
	}

	conn, err := net.Dial(network, url)
	if err != nil {
		return awserr.New("UDPError", "Could not connect", err)
	}

	rep.conn = conn

	return nil
}

func (rep *Reporter) close() {
	if rep.done != nil {
		close(rep.done)
	}

	rep.metricsCh.Pause()
}

func (rep *Reporter) start() {
	defer func() {
		rep.metricsCh.Pause()
	}()

	for {
		select {
		case <-rep.done:
			rep.done = nil
			return
		case m := <-rep.metricsCh.ch:
			// TODO: What to do with this error? Probably should just log
			b, err := json.Marshal(m)
			if err != nil {
				continue
			}

			rep.conn.Write(b)
		}
	}
}

// Pause will pause the metric channel preventing any new metrics from being
// added. It is safe to call concurrently with other calls to Pause, but if
// called concurently with Continue can lead to unexpected state.
func (rep *Reporter) Pause() {
	lock.Lock()
	defer lock.Unlock()

	if rep == nil {
		return
	}

	rep.close()
}

// Continue will reopen the metric channel and allow for monitoring to be
// resumed. It is safe to call concurrently with other calls to Continue, but
// if called concurently with Pause can lead to unexpected state.
func (rep *Reporter) Continue() {
	lock.Lock()
	defer lock.Unlock()
	if rep == nil {
		return
	}

	if !rep.metricsCh.IsPaused() {
		return
	}

	rep.metricsCh.Continue()
}

// Client side metric handler names
const (
	APICallMetricHandlerName        = "awscsm.SendAPICallMetric"
	APICallAttemptMetricHandlerName = "awscsm.SendAPICallAttemptMetric"
)

// InjectHandlers will will enable client side metrics and inject the proper
// handlers to handle how metrics are sent.
//
// InjectHandlers is NOT safe to call concurrently. Calling InjectHandlers
// multiple times may lead to unexpected behavior, (e.g. duplicate metrics).
//
//		// Start must be called in order to inject the correct handlers
//		r, err := csm.Start("clientID", "127.0.0.1:8094")
//		if err != nil {
//			panic(fmt.Errorf("expected no error, but received %v", err))
//		}
//
//		sess := session.NewSession()
//		r.InjectHandlers(&sess.Handlers)
//
//		// create a new service client with our client side metric session
//		svc := s3.New(sess)
func (rep *Reporter) InjectHandlers(handlers *request.Handlers) {
	if rep == nil {
		return
	}

	handlers.Complete.PushFrontNamed(request.NamedHandler{
		Name: APICallMetricHandlerName,
		Fn:   rep.sendAPICallMetric,
	})

	handlers.CompleteAttempt.PushFrontNamed(request.NamedHandler{
		Name: APICallAttemptMetricHandlerName,
		Fn:   rep.sendAPICallAttemptMetric,
	})
}

// boolIntValue return 1 for true and 0 for false.
func boolIntValue(b bool) int {
	if b {
		return 1
	}

	return 0
}
