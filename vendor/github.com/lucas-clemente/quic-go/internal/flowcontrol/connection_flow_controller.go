package flowcontrol

import (
	"errors"
	"fmt"
	"time"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/qerr"
	"github.com/lucas-clemente/quic-go/internal/utils"
)

type connectionFlowController struct {
	baseFlowController

	queueWindowUpdate func()
}

var _ ConnectionFlowController = &connectionFlowController{}

// NewConnectionFlowController gets a new flow controller for the connection
// It is created before we receive the peer's transport paramenters, thus it starts with a sendWindow of 0.
func NewConnectionFlowController(
	receiveWindow protocol.ByteCount,
	maxReceiveWindow protocol.ByteCount,
	queueWindowUpdate func(),
	rttStats *utils.RTTStats,
	logger utils.Logger,
) ConnectionFlowController {
	return &connectionFlowController{
		baseFlowController: baseFlowController{
			rttStats:             rttStats,
			receiveWindow:        receiveWindow,
			receiveWindowSize:    receiveWindow,
			maxReceiveWindowSize: maxReceiveWindow,
			logger:               logger,
		},
		queueWindowUpdate: queueWindowUpdate,
	}
}

func (c *connectionFlowController) SendWindowSize() protocol.ByteCount {
	return c.baseFlowController.sendWindowSize()
}

// IncrementHighestReceived adds an increment to the highestReceived value
func (c *connectionFlowController) IncrementHighestReceived(increment protocol.ByteCount) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.highestReceived += increment
	if c.checkFlowControlViolation() {
		return &qerr.TransportError{
			ErrorCode:    qerr.FlowControlError,
			ErrorMessage: fmt.Sprintf("received %d bytes for the connection, allowed %d bytes", c.highestReceived, c.receiveWindow),
		}
	}
	return nil
}

func (c *connectionFlowController) AddBytesRead(n protocol.ByteCount) {
	c.mutex.Lock()
	c.baseFlowController.addBytesRead(n)
	shouldQueueWindowUpdate := c.hasWindowUpdate()
	c.mutex.Unlock()
	if shouldQueueWindowUpdate {
		c.queueWindowUpdate()
	}
}

func (c *connectionFlowController) GetWindowUpdate() protocol.ByteCount {
	c.mutex.Lock()
	oldWindowSize := c.receiveWindowSize
	offset := c.baseFlowController.getWindowUpdate()
	if oldWindowSize < c.receiveWindowSize {
		c.logger.Debugf("Increasing receive flow control window for the connection to %d kB", c.receiveWindowSize/(1<<10))
	}
	c.mutex.Unlock()
	return offset
}

// EnsureMinimumWindowSize sets a minimum window size
// it should make sure that the connection-level window is increased when a stream-level window grows
func (c *connectionFlowController) EnsureMinimumWindowSize(inc protocol.ByteCount) {
	c.mutex.Lock()
	if inc > c.receiveWindowSize {
		c.logger.Debugf("Increasing receive flow control window for the connection to %d kB, in response to stream flow control window increase", c.receiveWindowSize/(1<<10))
		c.receiveWindowSize = utils.MinByteCount(inc, c.maxReceiveWindowSize)
		c.startNewAutoTuningEpoch(time.Now())
	}
	c.mutex.Unlock()
}

// The flow controller is reset when 0-RTT is rejected.
// All stream data is invalidated, it's if we had never opened a stream and never sent any data.
// At that point, we only have sent stream data, but we didn't have the keys to open 1-RTT keys yet.
func (c *connectionFlowController) Reset() error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if c.bytesRead > 0 || c.highestReceived > 0 || !c.epochStartTime.IsZero() {
		return errors.New("flow controller reset after reading data")
	}
	c.bytesSent = 0
	c.lastBlockedAt = 0
	return nil
}
