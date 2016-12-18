// +build linux

package libipvs

import (
	"encoding/hex"
	"fmt"
	"log"
	"syscall"

	"github.com/hkwi/nlgo"
)

type IPVSHandle interface {
	Flush() error
	GetInfo() (info Info, err error)
	ListServces() (services []*Service, err error)
	NewService(s *Service) error
	UpdateService(s *Service) error
	DelService(s *Service) error
	ListDestinations(s *Service) (dsts []*Destination, err error)
	NewDestination(s *Service, d *Destination) error
	UpdateDestination(s *Service, d *Destination) error
	DelDestination(s *Service, d *Destination) error
}

// Handle provides a ipvs handle to program ipvs rules.
type Handle struct {
	genlHub    *nlgo.GenlHub
	genlFamily nlgo.GenlFamily
}

// ResponseHandler know how to process netlink response
type ResponseHandler struct {
	Policy nlgo.MapPolicy
	Handle func(attrs nlgo.AttrMap) error
}

// New provides a new ipvs handle. It will return a valid handle or an error in case an
// error occurred while creating the handle.
func New() (IPVSHandle, error) {
	h := &Handle{}

	if genlHub, err := nlgo.NewGenlHub(); err != nil {
		return nil, err
	} else {
		h.genlHub = genlHub
	}
	// lookup family
	if genlFamily := h.genlHub.Family(IPVS_GENL_NAME); genlFamily.Id == 0 {
		return nil, fmt.Errorf("Invalid genl family: %v", IPVS_GENL_NAME)
	} else if genlFamily.Version != IPVS_GENL_VERSION {
		return nil, fmt.Errorf("Unsupported ipvs genl family: %+v", genlFamily)
	} else {
		h.genlFamily = genlFamily
	}
	return h, nil
}

var emptyAttrs = nlgo.AttrSlice{}

func (i *Handle) Flush() error {
	return i.doCmd(IPVS_CMD_FLUSH, syscall.NLM_F_ACK, emptyAttrs, nil)
}

func (i *Handle) ListServces() (services []*Service, err error) {
	respHandler := &ResponseHandler{
		Policy: ipvs_cmd_policy,
		Handle: func(attrs nlgo.AttrMap) error {
			if serviceAttrs := attrs.Get(IPVS_CMD_ATTR_SERVICE); serviceAttrs == nil {
				return fmt.Errorf("IPVS_CMD_GET_SERVICE without IPVS_CMD_ATTR_SERVICE")
			} else if service, err := unpackService(serviceAttrs.(nlgo.AttrMap)); err != nil {
				return err
			} else {
				services = append(services, &service)
			}
			return nil
		},
	}
	return services, i.doCmd(IPVS_CMD_GET_SERVICE, syscall.NLM_F_DUMP, emptyAttrs, respHandler)
}

func (i *Handle) ListDestinations(s *Service) (dsts []*Destination, err error) {
	respHandler := &ResponseHandler{
		Policy: ipvs_cmd_policy,
		Handle: func(attrs nlgo.AttrMap) error {
			if destAttrs := attrs.Get(IPVS_CMD_ATTR_DEST); destAttrs == nil {
				return fmt.Errorf("IPVS_CMD_GET_DEST without IPVS_CMD_ATTR_DEST")
			} else if dst, err := unpackDest(*s, destAttrs.(nlgo.AttrMap)); err != nil {
				return err
			} else {
				dsts = append(dsts, &dst)
			}
			return nil
		},
	}
	attrs := i.fillAttrs(s, nil, false, false)
	return dsts, i.doCmd(IPVS_CMD_GET_DEST, syscall.NLM_F_DUMP, attrs, respHandler)
}

func (i *Handle) GetInfo() (info Info, err error) {
	respHandler := &ResponseHandler{
		Policy: ipvs_info_policy,
		Handle: func(attrs nlgo.AttrMap) error {
			if cmdInfo, err := unpackInfo(attrs); err != nil {
				return err
			} else {
				info = cmdInfo
			}
			return nil
		},
	}
	return info, i.doCmd(IPVS_CMD_GET_INFO, syscall.NLM_F_ACK, emptyAttrs, respHandler)
}

// NewService creates a new ipvs service in the passed handle.
func (i *Handle) NewService(s *Service) error {
	attrs := i.fillAttrs(s, nil, true, false)
	return i.doCmd(IPVS_CMD_NEW_SERVICE, syscall.NLM_F_ACK, attrs, nil)
}

// UpdateService updates an already existing service in the passed
// handle.
func (i *Handle) UpdateService(s *Service) error {
	attrs := i.fillAttrs(s, nil, true, false)
	return i.doCmd(IPVS_CMD_SET_SERVICE, syscall.NLM_F_ACK, attrs, nil)
}

// DelService deletes an already existing service in the passed
// handle.
func (i *Handle) DelService(s *Service) error {
	attrs := i.fillAttrs(s, nil, false, false)
	return i.doCmd(IPVS_CMD_DEL_SERVICE, syscall.NLM_F_ACK, attrs, nil)
}

// NewDestination creates a new real server in the passed ipvs
// service which should already be existing in the passed handle.
func (i *Handle) NewDestination(s *Service, d *Destination) error {
	attrs := i.fillAttrs(s, d, false, true)
	return i.doCmd(IPVS_CMD_NEW_DEST, syscall.NLM_F_ACK, attrs, nil)
}

// UpdateDestination updates an already existing real server in the
// passed ipvs service in the passed handle.
func (i *Handle) UpdateDestination(s *Service, d *Destination) error {
	attrs := i.fillAttrs(s, d, false, true)
	return i.doCmd(IPVS_CMD_SET_DEST, syscall.NLM_F_ACK, attrs, nil)
}

// DelDestination deletes an already existing real server in the
// passed ipvs service in the passed handle.
func (i *Handle) DelDestination(s *Service, d *Destination) error {
	attrs := i.fillAttrs(s, d, false, false)
	return i.doCmd(IPVS_CMD_DEL_DEST, syscall.NLM_F_ACK, attrs, nil)
}

func (i *Handle) doCmd(cmd uint8, reqType uint16, attrs nlgo.AttrSlice, respHandler *ResponseHandler) error {
	req := i.genlFamily.Request(cmd, reqType, nil, attrs.Bytes())
	resp, err := i.genlHub.Sync(req)
	if err != nil {
		return err
	}

	for _, msg := range resp {
		if msg.Header.Type == syscall.NLMSG_ERROR {
			if msgErr := nlgo.NlMsgerr(msg.NetlinkMessage); msgErr.Payload().Error != 0 {
				return msgErr
			} else {
				// ack
			}
		} else if msg.Header.Type == syscall.NLMSG_DONE {
			// ack

		} else if msg.Family == i.genlFamily {
			if respHandler != nil {
				if attrsValue, err := respHandler.Policy.Parse(msg.Body()); err != nil {
					return fmt.Errorf("ipvs:Client.request: Invalid response: %s\n%s", err, hex.Dump(msg.Data))
				} else if attrMap, ok := attrsValue.(nlgo.AttrMap); !ok {
					return fmt.Errorf("ipvs:Client.request: Invalid attrs value: %v", attrsValue)
				} else {
					if err := respHandler.Handle(attrMap); err != nil {
						return err
					}
				}
			}
		} else {
			log.Printf("Client.request: Unknown response: %+v", msg)
		}
	}

	return nil
}

func (i *Handle) fillAttrs(s *Service, d *Destination, sfull, dfull bool) nlgo.AttrSlice {
	attrs := nlgo.AttrSlice{}
	if s != nil {
		attrs = append(attrs, nlattr(IPVS_CMD_ATTR_SERVICE, s.attrs(sfull)))
	}
	if d != nil {
		attrs = append(attrs, nlattr(IPVS_CMD_ATTR_DEST, d.attrs(s, dfull)))
	}
	return attrs
}
