package libstorage

import "github.com/emccode/libstorage/api/types"

type lss struct {
	types.Store
}

func (s *lss) GetServiceInfo(service string) *types.ServiceInfo {
	if obj, ok := s.Get(service).(*types.ServiceInfo); ok {
		return obj
	}
	return nil
}

func (s *lss) GetExecutorInfo(lsx string) *types.ExecutorInfo {
	if obj, ok := s.Get(lsx).(*types.ExecutorInfo); ok {
		return obj
	}
	return nil
}

func (s *lss) GetInstanceID(service string) *types.InstanceID {
	if obj, ok := s.Get(service).(*types.InstanceID); ok {
		return obj
	}
	return nil
}

func (s *lss) GetLocalDevices(service string) map[string]string {
	if obj, ok := s.Get(service).(map[string]string); ok {
		return obj
	}
	return nil
}
