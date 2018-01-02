package alert

import (
	"encoding/json"
	"fmt"
	"github.com/libopenstorage/openstorage/api"
	"github.com/portworx/kvdb"
	"go.pedge.io/dlog"
	"github.com/libopenstorage/openstorage/pkg/proto/time"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	// Name of this alert client implementation.
	Name = "alert_kvdb"
	// NameTest of this alert instance used only for unit tests.
	NameTest = "alert_kvdb_test"

	alertKey         = "alert/"
	subscriptionsKey = "subscriptions"
	nextAlertIDKey   = "nextAlertId"
	clusterKey       = "cluster/"
	volumeKey        = "volume/"
	nodeKey          = "node/"
	driveKey         = "drive/"
	lockKey          = "lock/"
	bootstrap        = "bootstrap"
	watchRetries     = 5
	watchSleep       = 100
)

const (
	watchBootstrap watcherStatus = iota
	watchReady
	watchError
)

var (
	kvdbMap     = make(map[string]kvdb.Kvdb)
	watcherMap  = make(map[string]*watcher)
	watchErrors int
	kvdbLock    sync.RWMutex
)

func init() {
	Register(Name, Init)
	Register(NameTest, Init)
}

type watcherStatus int

type watcher struct {
	kvcb      kvdb.WatchCB
	status    watcherStatus
	cb        AlertWatcherFunc
	clusterID string
	kvdb      kvdb.Kvdb
}

// KvAlert is used for managing the alerts and its kvdb instance
type KvAlert struct {
	// clusterID for which this alerts object will be used
	clusterID string
}

func getLockId(resourceId, uniqueTag string) string {
	return lockKey + resourceId + "." + uniqueTag + ".lock"
}

// GetKvdbInstance returns a kvdb instance associated with this alert client and clusterID combination.
func (kva *KvAlert) GetKvdbInstance() kvdb.Kvdb {
	kvdbLock.RLock()
	defer kvdbLock.RUnlock()
	return kvdbMap[kva.clusterID]
}

// Init initializes a AlertClient interface implementation.
func Init(kv kvdb.Kvdb, clusterID string) (Alert, error) {
	kvdbLock.Lock()
	defer kvdbLock.Unlock()
	if _, ok := kvdbMap[clusterID]; !ok {
		kvdbMap[clusterID] = kv
	}
	return &KvAlert{clusterID}, nil
}

// Raise raises an Alert.
func (kva *KvAlert) Raise(a *api.Alert) error {
	var subscriptions []api.Alert
	kv := kva.GetKvdbInstance()
	if _, err := kv.GetVal(getSubscriptionsKey(a.AlertType), &subscriptions); err != nil {
		if err != kvdb.ErrNotFound {
			return err
		}
	} else {
		for _, alert := range subscriptions {
			if err := kva.Raise(&alert); err != nil {
				return ErrSubscribedRaise
			}
		}
	}
	return kva.raise(a)
}

// Raise raises an Alert if does not exists yet.
func (kva *KvAlert) RaiseIfNotExist(a *api.Alert) error {
	if strings.TrimSpace(a.ResourceId) == "" ||
		strings.TrimSpace(a.UniqueTag) == "" {
		return ErrIllegal
	}
	var subscriptions []api.Alert
	kv := kva.GetKvdbInstance()
	if _, err := kv.GetVal(getSubscriptionsKey(a.AlertType), &subscriptions); err != nil {
		if err != kvdb.ErrNotFound {
			return err
		}
	} else {
		for _, alert := range subscriptions {
			if err := kva.RaiseIfNotExist(&alert); err != nil {
				return ErrSubscribedRaise
			}
		}
	}
	return kva.raiseIfNotExist(a)
}

// Subscribe allows a child (dependent) alert to subscribe to a parent alert
func (kva *KvAlert) Subscribe(parentAlertType int64, childAlert *api.Alert) error {
	var subscriptions []api.Alert
	kv := kva.GetKvdbInstance()
	if _, err := kv.GetVal(getSubscriptionsKey(parentAlertType), &subscriptions); err != nil {
		if err != kvdb.ErrNotFound {
			return err
		}
	}
	subscriptions = append(subscriptions, *childAlert)
	_, err := kv.Put(getSubscriptionsKey(parentAlertType), subscriptions, 0)
	return err
}

// Erase erases an alert.
func (kva *KvAlert) Erase(resourceType api.ResourceType, alertID int64) error {
	kv := kva.GetKvdbInstance()
	if resourceType == api.ResourceType_RESOURCE_TYPE_NONE {
		return ErrResourceNotFound
	}
	_, err := kv.Delete(getResourceKey(resourceType) + strconv.FormatInt(alertID, 10))
	return err
}

// Clear clears an alert.
func (kva *KvAlert) Clear(resourceType api.ResourceType, alertID int64, ttl uint64) error {
	return kva.clear(resourceType, alertID, ttl)
}

// Retrieve retrieves a specific alert.
func (kva *KvAlert) Retrieve(resourceType api.ResourceType, alertID int64) (*api.Alert, error) {
	var alert api.Alert
	if resourceType == api.ResourceType_RESOURCE_TYPE_NONE {
		return &alert, ErrResourceNotFound
	}
	kv := kva.GetKvdbInstance()
	_, err := kv.GetVal(getResourceKey(resourceType)+strconv.FormatInt(alertID, 10), &alert)
	return &alert, err
}

// Enumerate enumerates alert
func (kva *KvAlert) Enumerate(filter *api.Alert) ([]*api.Alert, error) {
	kv := kva.GetKvdbInstance()
	return kva.enumerate(kv, filter)
}

// EnumerateWithinTimeRange enumerates alert between timeStart and timeEnd.
func (kva *KvAlert) EnumerateWithinTimeRange(
	timeStart time.Time,
	timeEnd time.Time,
	resourceType api.ResourceType,
) ([]*api.Alert, error) {
	allAlerts := []*api.Alert{}
	resourceAlerts := []*api.Alert{}
	var err error

	kv := kva.GetKvdbInstance()
	if resourceType != 0 {
		resourceAlerts, err = kva.getResourceSpecificAlerts(resourceType, kv)
		if err != nil {
			return nil, err
		}
	} else {
		resourceAlerts, err = kva.getAllAlerts(kv)
		if err != nil {
			return nil, err
		}
	}
	for _, v := range resourceAlerts {
		alertTime := prototime.TimestampToTime(v.Timestamp)
		if alertTime.Before(timeEnd) && alertTime.After(timeStart) {
			allAlerts = append(allAlerts, v)
		}
	}
	return allAlerts, nil
}

// Watch on all Alerts for the given clusterID. It uses the global
// kvdb options provided while creating the alertClient object to access this cluster
// This way we ensure that the caller of the api is able to watch alerts on clusters that
// it is authorized for.
func (kva *KvAlert) Watch(clusterID string, alertWatcherFunc AlertWatcherFunc) error {

	kv, err := kva.getKvdbForCluster(clusterID)
	if err != nil {
		return err
	}

	alertWatcher := &watcher{status: watchBootstrap, cb: alertWatcherFunc, kvcb: kvdbWatch, kvdb: kv}
	watcherKey := clusterID
	watcherMap[watcherKey] = alertWatcher

	if err := subscribeWatch(watcherKey); err != nil {
		return err
	}

	// Subscribe for a watch can be in a goroutine. Bootstrap by writing to the key and waiting for an update
	retries := 0

	for alertWatcher.status == watchBootstrap {
		if _, err := kv.Put(alertKey+bootstrap, time.Now(), 0); err != nil {
			return err
		}
		if alertWatcher.status == watchBootstrap {
			retries++
			// TODO(pedge): constant, maybe configurable
			time.Sleep(time.Millisecond * watchSleep)
		}
		// TODO(pedge): constant, maybe configurable
		if retries == watchRetries {
			return fmt.Errorf("Failed to bootstrap watch on %s", clusterID)
		}
	}
	if alertWatcher.status != watchReady {
		return fmt.Errorf("Failed to watch on %s", clusterID)
	}
	return nil
}

// Shutdown shutdown
func (kva *KvAlert) Shutdown() {
}

// String
func (kva *KvAlert) String() string {
	return Name
}

func getResourceKey(resourceType api.ResourceType) string {
	if resourceType == api.ResourceType_RESOURCE_TYPE_VOLUME {
		return alertKey + volumeKey
	}
	if resourceType == api.ResourceType_RESOURCE_TYPE_NODE {
		return alertKey + nodeKey
	}
	if resourceType == api.ResourceType_RESOURCE_TYPE_CLUSTER {
		return alertKey + clusterKey
	}
	return alertKey + driveKey
}

func getNextAlertIDKey() string {
	return alertKey + nextAlertIDKey
}

func getSubscriptionsKey(alertType int64) string {
	return alertKey + subscriptionsKey + "/" + strconv.FormatInt(alertType, 10)
}

func (kva *KvAlert) raise(a *api.Alert) error {
	kv := kva.GetKvdbInstance()
	if a.Resource == api.ResourceType_RESOURCE_TYPE_NONE {
		return ErrResourceNotFound
	}
	alertID, err := kva.getNextIDFromKVDB()
	if err != nil {
		return err
	}
	// TODO(pedge): when this is changed to a pointer, we need to rethink this.
	a.Id = alertID
	a.Timestamp = prototime.Now()
	a.Cleared = false
	_, err = kv.Create(getResourceKey(a.Resource)+strconv.FormatInt(a.Id, 10), a, a.Ttl)
	return err

}

func (kva *KvAlert) raiseIfNotExist(a *api.Alert) error {
	kv := kva.GetKvdbInstance()
	if a.Resource == api.ResourceType_RESOURCE_TYPE_NONE {
		return ErrResourceNotFound
	}

	// Acquire resource lock: lockKey/resouceId.uniqueTag.lock.
	// This ensures only one raiseIfNotExists operation for a given resource
	// is able to proceed.
	kvp, err := kv.Lock(getLockId(a.ResourceId, a.UniqueTag))
	if err != nil {
		dlog.Errorf("Failed to get lock for resource %s, err: %s",
			a.ResourceId, err.Error())
		return err
	}
	defer kv.Unlock(kvp)

	alerts, err := kva.getResourceSpecificAlerts(a.Resource, kv)
	if err != nil {
		dlog.Infof("Failed to get alerts of type %s, error: %s",
			a.Resource, err.Error())
		return err
	}
	for _, alert := range alerts {
		if alert.ResourceId == a.ResourceId && alert.UniqueTag == a.UniqueTag {
			a.Id = alert.Id
			return nil
		}
	}

	// Alert does ot exist, raise a new one
	return kva.raise(a)
}

func (kva *KvAlert) ClearByUniqueTag(
	resourceType api.ResourceType,
	resourceId string,
	uniqueTag string,
	ttl uint64,
) error {
	kv := kva.GetKvdbInstance()
	if resourceType == api.ResourceType_RESOURCE_TYPE_NONE {
		return ErrResourceNotFound
	}
	if uniqueTag == "" || resourceId == "" {
		return ErrIllegal
	}

	kvp, err := kv.Lock(getLockId(resourceId, uniqueTag))
	if err != nil {
		dlog.Errorf("Failed to get lock for resource %s, err: %s",
			resourceId, err.Error())
		return err
	}
	defer kv.Unlock(kvp)

	alerts, err := kva.getResourceSpecificAlerts(resourceType, kv)
	if err != nil {
		dlog.Infof("Failed to get alerts of type %s, error: %s",
			resourceType, err.Error())
		return err
	}
	for _, alert := range alerts {
		if resourceId == alert.ResourceId && uniqueTag == alert.UniqueTag {
			return kva.clear(resourceType, alert.Id, ttl)
		}
	}

	// Alert does ot exist, return
	return nil
}

func (kva *KvAlert) clear(resourceType api.ResourceType, alertID int64, ttl uint64) error {
	kv := kva.GetKvdbInstance()
	var alert api.Alert
	if resourceType == api.ResourceType_RESOURCE_TYPE_NONE {
		return ErrResourceNotFound
	}
	if _, err := kv.GetVal(getResourceKey(resourceType)+strconv.FormatInt(alertID, 10), &alert); err != nil {
		return err
	}
	alert.Cleared = true

	_, err := kv.Update(getResourceKey(resourceType)+strconv.FormatInt(alertID, 10), &alert, ttl)
	return err
}

func (kva *KvAlert) getNextIDFromKVDB() (int64, error) {
	kv := kva.GetKvdbInstance()
	nextAlertID := 0
	kvp, err := kv.Create(getNextAlertIDKey(), strconv.FormatInt(int64(nextAlertID+1), 10), 0)

	for err != nil {
		kvp, err = kv.GetVal(getNextAlertIDKey(), &nextAlertID)
		if err != nil {
			err = ErrNotInitialized
			return -1, err
		}
		prevValue := kvp.Value
		newKvp := *kvp
		newKvp.Value = []byte(strconv.FormatInt(int64(nextAlertID+1), 10))
		kvp, err = kv.CompareAndSet(&newKvp, kvdb.KVFlags(0), prevValue)
	}
	return int64(nextAlertID), err
}

func (kva *KvAlert) getResourceSpecificAlerts(resourceType api.ResourceType, kv kvdb.Kvdb) ([]*api.Alert, error) {
	allAlerts := []*api.Alert{}
	kvp, err := kv.Enumerate(getResourceKey(resourceType))
	if err != nil {
		return nil, err
	}

	for _, v := range kvp {
		var elem *api.Alert
		if err := json.Unmarshal(v.Value, &elem); err != nil {
			return nil, err
		}
		allAlerts = append(allAlerts, elem)
	}
	return allAlerts, nil
}

func (kva *KvAlert) getAllAlerts(kv kvdb.Kvdb) ([]*api.Alert, error) {
	allAlerts := []*api.Alert{}
	clusterAlerts := []*api.Alert{}
	nodeAlerts := []*api.Alert{}
	volumeAlerts := []*api.Alert{}
	driveAlerts := []*api.Alert{}
	var err error

	nodeAlerts, err = kva.getResourceSpecificAlerts(api.ResourceType_RESOURCE_TYPE_NODE, kv)
	if err == nil {
		allAlerts = append(allAlerts, nodeAlerts...)
	}
	volumeAlerts, err = kva.getResourceSpecificAlerts(api.ResourceType_RESOURCE_TYPE_VOLUME, kv)
	if err == nil {
		allAlerts = append(allAlerts, volumeAlerts...)
	}
	clusterAlerts, err = kva.getResourceSpecificAlerts(api.ResourceType_RESOURCE_TYPE_CLUSTER, kv)
	if err == nil {
		allAlerts = append(allAlerts, clusterAlerts...)
	}
	driveAlerts, err = kva.getResourceSpecificAlerts(api.ResourceType_RESOURCE_TYPE_DRIVE, kv)
	if err == nil {
		allAlerts = append(allAlerts, driveAlerts...)
	}

	if len(allAlerts) > 0 {
		return allAlerts, nil
	} else if len(allAlerts) == 0 {
		return nil, fmt.Errorf("No alert raised yet")
	}
	return allAlerts, err
}

func (kva *KvAlert) enumerate(kv kvdb.Kvdb, filter *api.Alert) ([]*api.Alert, error) {
	allAlerts := []*api.Alert{}
	resourceAlerts := []*api.Alert{}
	var err error

	if filter.Resource != api.ResourceType_RESOURCE_TYPE_NONE {
		resourceAlerts, err = kva.getResourceSpecificAlerts(filter.Resource, kv)
		if err != nil {
			return nil, err
		}
	} else {
		resourceAlerts, err = kva.getAllAlerts(kv)
	}

	if filter.Severity != 0 {
		for _, v := range resourceAlerts {
			if v.Severity <= filter.Severity {
				allAlerts = append(allAlerts, v)
			}
		}
	} else {
		allAlerts = append(allAlerts, resourceAlerts...)
	}

	return allAlerts, err
}

func (kva *KvAlert) getKvdbForCluster(clusterID string) (kvdb.Kvdb, error) {
	kvdbLock.Lock()
	defer kvdbLock.Unlock()

	kv, ok := kvdbMap[clusterID]
	if !ok {
		return nil, fmt.Errorf("Unknown cluster ID %v", clusterID)
	}
	return kv, nil
}

func kvdbWatch(prefix string, opaque interface{}, kvp *kvdb.KVPair, err error) error {
	lock.Lock()
	defer lock.Unlock()

	watcherKey := strings.Split(prefix, "/")[1]

	if err == nil && strings.HasSuffix(kvp.Key, bootstrap) {
		w := watcherMap[watcherKey]
		w.status = watchReady
		return nil
	}

	if err != nil {
		if w := watcherMap[watcherKey]; w.status == watchBootstrap {
			w.status = watchError
			return err
		}
		if watchErrors == 5 {
			w := watcherMap[watcherKey]
			dlog.Warnf("Too many watch errors for key (%v). Error: %s. Stopping the watch!!", watcherKey, err.Error())
			w.cb(nil, api.AlertActionType_ALERT_ACTION_TYPE_NONE, prefix, "")
			// Too many watch errors. Stop the watch
			return err
		}
		watchErrors++
		if err := subscribeWatch(watcherKey); err != nil {
			dlog.Warnf("Failed to resubscribe : %s", err.Error())
		}
		return err
	}

	if strings.HasSuffix(kvp.Key, nextAlertIDKey) || strings.Contains(kvp.Key, subscriptionsKey) {
		// Ignore write on this key
		// Todo : Add a map of ignore keys
		return nil
	}
	watchErrors = 0

	w := watcherMap[watcherKey]

	if kvp.Action == kvdb.KVDelete {
		err = w.cb(nil, api.AlertActionType_ALERT_ACTION_TYPE_DELETE, prefix, kvp.Key)
		return err
	}

	var alert api.Alert
	if err := json.Unmarshal(kvp.Value, &alert); err != nil {
		return fmt.Errorf("Failed to unmarshal Alert")
	}

	switch kvp.Action {
	case kvdb.KVCreate:
		err = w.cb(&alert, api.AlertActionType_ALERT_ACTION_TYPE_CREATE, prefix, kvp.Key)
	case kvdb.KVSet:
		err = w.cb(&alert, api.AlertActionType_ALERT_ACTION_TYPE_UPDATE, prefix, kvp.Key)
	default:
		err = fmt.Errorf("Unhandled KV Action")
	}
	return err
}

func subscribeWatch(key string) error {
	// Always set the watchIndex to 0
	watchIndex := 0
	w, ok := watcherMap[key]
	if !ok {
		return fmt.Errorf("Failed to find a watch on cluster : %v", key)
	}

	kv := w.kvdb
	if err := kv.WatchTree(alertKey, uint64(watchIndex), nil, w.kvcb); err != nil {
		return err
	}
	return nil
}
