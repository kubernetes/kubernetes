package master

import (
	"net"
	"testing"
)

func TestServiceIPRange(t *testing.T) {
	type args struct {
		passedServiceClusterIPRange net.IPNet
		passedAPIServiceClusterIP   net.IP
	}
	tests := []struct {
		name                    string
		args                    args
		serviceRange, serviceIP string
		wantErr                 bool
	}{
		{
			name: "empty service cluster IP should fetch the first IP from the range",
			args: args{
				passedServiceClusterIPRange: net.IPNet{IP: net.IPv4(10, 96, 0, 0), Mask: net.CIDRMask(16, 32)},
			},
			serviceRange: "10.96.0.0/16",
			serviceIP:    "10.96.0.1",
			wantErr:      false,
		},
		{
			name: "custom service cluster IP should just return it",
			args: args{
				passedServiceClusterIPRange: net.IPNet{IP: net.IPv4(10, 96, 0, 0), Mask: net.CIDRMask(16, 32)},
				passedAPIServiceClusterIP:   net.IPv4(10, 96, 0, 5),
			},
			serviceRange: "10.96.0.0/16",
			serviceIP:    "10.96.0.5",
			wantErr:      false,
		},
		{
			name: "custom service cluster IP not in service IP range should return error",
			args: args{
				passedServiceClusterIPRange: net.IPNet{IP: net.IPv4(10, 96, 0, 0), Mask: net.CIDRMask(16, 32)},
				passedAPIServiceClusterIP:   net.IPv4(10, 91, 0, 5),
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			serviceRange, serviceIP, err := ServiceIPRange(tt.args.passedServiceClusterIPRange, tt.args.passedAPIServiceClusterIP)
			if err != nil {
				if !tt.wantErr {
					t.Errorf("ServiceIPRange() error = %v, wantErr %v", err, tt.wantErr)
				}
				return
			}
			if serviceRange.String() != tt.serviceRange {
				t.Errorf("ServiceIPRange() got = %v, want %s", serviceRange, tt.serviceRange)
			}
			if serviceIP.String() != tt.serviceIP {
				t.Errorf("ServiceIPRange() got1 = %v, want %s", serviceIP, tt.serviceIP)
			}
		})
	}
}
