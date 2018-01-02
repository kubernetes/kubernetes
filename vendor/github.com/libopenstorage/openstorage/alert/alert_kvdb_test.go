package alert

import (
	"fmt"
	"github.com/libopenstorage/openstorage/api"
	"github.com/portworx/kvdb"
	"github.com/portworx/kvdb/mem"
	"github.com/stretchr/testify/require"
	"go.pedge.io/dlog"
	"github.com/libopenstorage/openstorage/pkg/proto/time"
	"strconv"
	"sync"
	"testing"
	"time"
)

var (
	kva             Alert
	nextID          int64
	isWatcherCalled int
	watcherAction   api.AlertActionType
	watcherAlert    api.Alert
	watcherPrefix   string
	watcherKey      string
)

const (
	kvdbDomain     = "openstorage"
	clusterName    = "1"
	newClusterName = "2"
)

func TestAll(t *testing.T) {
	setup(t)
	raiseAndErase(t)
	raiseWithTTL(t)
	subscribe(t)
	retrieve(t)
	clear(t)
	clearWithTTL(t)
	enumerate(t)
	watch(t)
}

func setup(t *testing.T) {
	kv := kvdb.Instance()
	if kv == nil {
		kv, err := kvdb.New(mem.Name, kvdbDomain+"/"+clusterName, []string{}, nil, dlog.Panicf)
		if err != nil {
			t.Fatalf("Failed to set default KV store : (%v): %v", mem.Name, err)
		}
		err = kvdb.SetInstance(kv)
		if err != nil {
			t.Fatalf("Failed to set default KV store: (%v): %v", mem.Name, err)
		}
	}

	var err error
	kva, err = New("alert_kvdb", clusterName, kvdb.Instance())
	if err != nil {
		t.Fatalf("Failed to create new Kvapi.Alert object")
	}
}

func raiseAndErase(t *testing.T) {
	// Raise api.Alert Id : 1
	raiseAlert := api.Alert{
		Resource: api.ResourceType_RESOURCE_TYPE_VOLUME,
		Severity: api.SeverityType_SEVERITY_TYPE_NOTIFY,
		Message:  "Test Message",
	}
	err := kva.Raise(&raiseAlert)
	require.NoError(t, err, "Failed in raising an alert")

	kv := kva.GetKvdbInstance()
	var alert api.Alert

	_, err = kv.GetVal(
		getResourceKey(api.ResourceType_RESOURCE_TYPE_VOLUME)+
			strconv.FormatInt(raiseAlert.Id, 10), &alert)
	require.NoError(t, err, "Failed to retrieve alert from kvdb")
	require.NotNil(t, alert, "api.Alert object null in kvdb")
	require.Equal(t, raiseAlert.Id, alert.Id, "api.Alert Id mismatch")
	require.Equal(t, api.ResourceType_RESOURCE_TYPE_VOLUME, alert.Resource, "api.Alert Resource mismatch")
	require.Equal(t, api.SeverityType_SEVERITY_TYPE_NOTIFY, alert.Severity, "api.Alert Severity mismatch")

	// Raise api.Alert with no Resource
	err = kva.Raise(&api.Alert{Severity: api.SeverityType_SEVERITY_TYPE_NOTIFY})
	require.Error(t, err, "An error was expected")
	require.Equal(t, ErrResourceNotFound, err, "Error mismatch")

	// Erase api.Alert Id : 1
	err = kva.Erase(api.ResourceType_RESOURCE_TYPE_VOLUME, raiseAlert.Id)
	require.NoError(t, err, "Failed to erase an alert")

	_, err = kv.GetVal(getResourceKey(api.ResourceType_RESOURCE_TYPE_VOLUME)+"1", &alert)
	require.Error(t, err, "api.Alert not erased from kvdb")
}

func raiseIfNotExistAndErase(t *testing.T) {
	alerts := make([]*api.Alert, 0)
	mtx := &sync.Mutex{}
	alertFn := func() {
		raiseAlert := &api.Alert{
			Resource:   api.ResourceType_RESOURCE_TYPE_VOLUME,
			Severity:   api.SeverityType_SEVERITY_TYPE_NOTIFY,
			Message:    "Test Message",
			ResourceId: "vol1",
			UniqueTag:  "alert_type_1",
		}
		err := kva.RaiseIfNotExist(raiseAlert)
		require.NoError(t, err, "Failed in raising an alert")
		mtx.Lock()
		defer mtx.Unlock()
		alerts = append(alerts, raiseAlert)
	}

	runs := 5
	for i := 0; i < runs; i++ {
		go alertFn()
	}

	require.Equal(t, len(alerts), runs, "alerts")
	for _, a := range alerts {
		require.Equal(t, a.Id, alerts[0].Id, "ids match")
	}

	// Raise an alert for same resource type but different id.
	alerts[1].UniqueTag = "alert_type_2"
	err := kva.RaiseIfNotExist(alerts[1])
	require.NoError(t, err, "Failed in raising an alert")
	require.NotEqual(t, alerts[1].Id, alerts[0].Id, "different resources")

	err = kva.ClearByUniqueTag(api.ResourceType_RESOURCE_TYPE_VOLUME,
		alerts[0].ResourceId, alerts[0].UniqueTag, 2000)
	require.NoError(t, err, "Failed to erase an alert")

	kv := kva.GetKvdbInstance()
	_, err = kv.GetVal(getResourceKey(api.ResourceType_RESOURCE_TYPE_VOLUME)+
		strconv.FormatInt(alerts[0].Id, 10), alerts[0])
	require.NoError(t, err, "api.Alert erased from kvdb")
	require.True(t, alerts[0].Cleared, "api.Alert erased from kvdb")

	err = kva.Erase(api.ResourceType_RESOURCE_TYPE_VOLUME, alerts[0].Id)
	require.NoError(t, err, "Failed to erase an alert")

	kv = kva.GetKvdbInstance()
	_, err = kv.GetVal(getResourceKey(api.ResourceType_RESOURCE_TYPE_VOLUME)+
		strconv.FormatInt(alerts[0].Id, 10), alerts[0])
	require.Error(t, err, "api.Alert not erased from kvdb")

	alerts[0].ResourceId = ""
	err = kva.RaiseIfNotExist(alerts[0])
	require.Error(t, err, "resource id missing check")

	alerts[0].UniqueTag = ""
	err = kva.RaiseIfNotExist(alerts[0])
	require.Error(t, err, "resource id missing check")

	// cleanup
	err = kva.Erase(api.ResourceType_RESOURCE_TYPE_VOLUME, alerts[1].Id)
	require.NoError(t, err, "Failed to erase an alert")

}

func raiseWithTTL(t *testing.T) {
	raiseAlert := api.Alert{
		Resource: api.ResourceType_RESOURCE_TYPE_VOLUME,
		Severity: api.SeverityType_SEVERITY_TYPE_NOTIFY,
		Message:  "Test Message",
		Ttl:      2,
	}
	err := kva.Raise(&raiseAlert)
	require.NoError(t, err, "Failed in raising an alert")

	kv := kva.GetKvdbInstance()
	var alert api.Alert

	_, err = kv.GetVal(getResourceKey(api.ResourceType_RESOURCE_TYPE_VOLUME)+strconv.FormatInt(raiseAlert.Id, 10), &alert)
	require.NoError(t, err, "Failed to retrieve alert from kvdb")
	require.NotNil(t, alert, "api.Alert object null in kvdb")

	time.Sleep(3 * time.Second)
	_, err = kv.GetVal(getResourceKey(api.ResourceType_RESOURCE_TYPE_VOLUME)+strconv.FormatInt(raiseAlert.Id, 10), &alert)
	require.Error(t, err, "Alert should have been removed after the ttl value: ", time.Now())
}

func subscribe(t *testing.T) {
	parentAlertType := int64(1)
	child1Alert := api.Alert{
		AlertType: 2,
		Message:   "child 1",
		Resource:  api.ResourceType_RESOURCE_TYPE_DRIVE,
		Severity:  api.SeverityType_SEVERITY_TYPE_NOTIFY,
	}
	child2Alert := api.Alert{
		AlertType: 3,
		Message:   "child 2",
		Resource:  api.ResourceType_RESOURCE_TYPE_VOLUME,
		Severity:  api.SeverityType_SEVERITY_TYPE_NOTIFY,
	}
	err := kva.Subscribe(parentAlertType, &child1Alert)
	require.NoError(t, err, "Failed to subscribe alert")
	err = kva.Subscribe(2, &child2Alert)
	require.NoError(t, err, "Failed to subscribe alert")

	raiseAlert := api.Alert{
		AlertType: parentAlertType,
		Resource:  api.ResourceType_RESOURCE_TYPE_NODE,
		Severity:  api.SeverityType_SEVERITY_TYPE_NOTIFY,
		Message:   "parent",
	}
	err = kva.Raise(&raiseAlert)
	require.NoError(t, err, "Failed to raise parent alert")

	enAlerts, err := kva.Enumerate(&api.Alert{})
	require.Equal(t, 3, len(enAlerts), "Incorrect number of alerts raised")

	for _, a := range enAlerts {
		err = kva.Erase(a.Resource, a.Id)
	}

	child3Alert := api.Alert{
		AlertType: 4,
		Message:   "child 3",
		Resource:  api.ResourceType_RESOURCE_TYPE_VOLUME,
		Severity:  api.SeverityType_SEVERITY_TYPE_NOTIFY,
	}
	err = kva.Subscribe(parentAlertType, &child3Alert)
	require.NoError(t, err, "Failed to subscribe alert")
	err = kva.Raise(&raiseAlert)
	require.NoError(t, err, "Failed to raise parent alert")

	enAlerts, err = kva.Enumerate(&api.Alert{})
	require.Equal(t, 4, len(enAlerts), "Incorrect number of alerts raised")

	for _, a := range enAlerts {
		err = kva.Erase(a.Resource, a.Id)
	}

}

func retrieve(t *testing.T) {
	var alert *api.Alert

	// Raise a ResourceType_RESOURCE_TYPE_NODE specific api.Alert
	raiseAlert := api.Alert{
		Resource: api.ResourceType_RESOURCE_TYPE_NODE,
		Severity: api.SeverityType_SEVERITY_TYPE_ALARM,
	}
	err := kva.Raise(&raiseAlert)
	fmt.Printf("Raise err : %s, Raise Id : %d \n", err, raiseAlert.Id)

	alert, err = kva.Retrieve(api.ResourceType_RESOURCE_TYPE_NODE, raiseAlert.Id)
	require.NoError(t, err, "Failed to retrieve alert")
	require.NotNil(t, alert, "api.Alert object null")
	require.Equal(t, raiseAlert.Id, alert.Id, "api.Alert Id mismatch")
	require.Equal(t, api.ResourceType_RESOURCE_TYPE_NODE, alert.Resource, "api.Alert resource mismatch")
	require.Equal(t, api.SeverityType_SEVERITY_TYPE_ALARM, alert.Severity, "api.Alert severity mismatch")

	// Retrieve non existing alert
	alert, err = kva.Retrieve(api.ResourceType_RESOURCE_TYPE_VOLUME, 5)
	require.Error(t, err, "Expected an error")

	// Cleanup
	err = kva.Erase(api.ResourceType_RESOURCE_TYPE_NODE, raiseAlert.Id)
}

func clear(t *testing.T) {
	// Raise an alert
	var alert api.Alert
	kv := kva.GetKvdbInstance()
	raiseAlert := api.Alert{
		Resource: api.ResourceType_RESOURCE_TYPE_NODE,
		Severity: api.SeverityType_SEVERITY_TYPE_ALARM,
	}
	err := kva.Raise(&raiseAlert)

	err = kva.Clear(api.ResourceType_RESOURCE_TYPE_NODE, raiseAlert.Id, 0)
	require.NoError(t, err, "Failed to clear alert")

	_, err = kv.GetVal(getResourceKey(api.ResourceType_RESOURCE_TYPE_NODE)+strconv.FormatInt(raiseAlert.Id, 10), &alert)
	require.Equal(t, true, alert.Cleared, "Failed to clear alert")

	err = kva.Erase(api.ResourceType_RESOURCE_TYPE_NODE, raiseAlert.Id)
}

func clearWithTTL(t *testing.T) {
	// Raise an alert
	var alert api.Alert
	kv := kva.GetKvdbInstance()
	raiseAlert := api.Alert{
		Resource: api.ResourceType_RESOURCE_TYPE_NODE,
		Severity: api.SeverityType_SEVERITY_TYPE_ALARM,
	}
	err := kva.Raise(&raiseAlert)

	err = kva.Clear(api.ResourceType_RESOURCE_TYPE_NODE, raiseAlert.Id, 2)
	require.NoError(t, err, "Failed to clear alert")

	_, err = kv.GetVal(getResourceKey(api.ResourceType_RESOURCE_TYPE_NODE)+strconv.FormatInt(raiseAlert.Id, 10), &alert)
	require.Equal(t, true, alert.Cleared, "Failed to clear alert")

	time.Sleep(3 * time.Second)
	_, err = kv.GetVal(getResourceKey(api.ResourceType_RESOURCE_TYPE_NODE)+strconv.FormatInt(raiseAlert.Id, 10), &alert)
	require.Error(t, err, "Cleared Alert should have been removed after the ttl value")
}

func enumerate(t *testing.T) {
	// Raise a few alert
	raiseAlert1 := api.Alert{
		Resource: api.ResourceType_RESOURCE_TYPE_VOLUME,
		Severity: api.SeverityType_SEVERITY_TYPE_NOTIFY,
	}
	err := kva.Raise(&raiseAlert1)
	raiseAlert2 := api.Alert{
		Resource: api.ResourceType_RESOURCE_TYPE_VOLUME,
		Severity: api.SeverityType_SEVERITY_TYPE_NOTIFY,
	}
	err = kva.Raise(&raiseAlert2)
	raiseAlert3 := api.Alert{
		Resource: api.ResourceType_RESOURCE_TYPE_VOLUME,
		Severity: api.SeverityType_SEVERITY_TYPE_WARNING,
	}
	err = kva.Raise(&raiseAlert3)
	raiseAlert4 := api.Alert{
		Resource: api.ResourceType_RESOURCE_TYPE_NODE,
		Severity: api.SeverityType_SEVERITY_TYPE_WARNING,
	}
	err = kva.Raise(&raiseAlert4)

	enAlerts, err := kva.Enumerate(&api.Alert{Resource: api.ResourceType_RESOURCE_TYPE_VOLUME})
	require.NoError(t, err, "Failed to enumerate alert")
	require.Equal(t, 3, len(enAlerts), "Enumerated incorrect number of alert")

	enAlerts, err = kva.Enumerate(&api.Alert{Resource: api.ResourceType_RESOURCE_TYPE_VOLUME, Severity: api.SeverityType_SEVERITY_TYPE_WARNING})
	require.NoError(t, err, "Failed to enumerate alert")
	require.Equal(t, 1, len(enAlerts), "Enumerated incorrect number of alert")
	require.Equal(t, api.SeverityType_SEVERITY_TYPE_WARNING, enAlerts[0].Severity, "Severity mismatch")

	enAlerts, err = kva.Enumerate(&api.Alert{})
	require.NoError(t, err, "Failed to enumerate alert")
	require.Equal(t, 4, len(enAlerts), "Enumerated incorrect number of alert")

	enAlerts, err = kva.Enumerate(&api.Alert{Severity: api.SeverityType_SEVERITY_TYPE_WARNING})
	require.NoError(t, err, "Failed to enumerate alert")
	require.Equal(t, 2, len(enAlerts), "Enumerated incorrect number of alert")
	require.Equal(t, api.SeverityType_SEVERITY_TYPE_WARNING, enAlerts[0].Severity, "Severity mismatch")

	// Add a dummy event into kvdb two hours ago
	kv := kva.GetKvdbInstance()
	currentTime := time.Now()
	delayedTime := currentTime.Add(-1 * time.Duration(2) * time.Hour)

	var fakeAlertId int64
	fakeAlertId = 100
	alert := api.Alert{Timestamp: prototime.TimeToTimestamp(delayedTime), Id: fakeAlertId, Resource: api.ResourceType_RESOURCE_TYPE_VOLUME}

	_, err = kv.Put(getResourceKey(api.ResourceType_RESOURCE_TYPE_VOLUME)+strconv.FormatInt(fakeAlertId, 10), &alert, 0)
	enAlerts, err = kva.EnumerateWithinTimeRange(currentTime.Add(-1*time.Duration(10)*time.Second), currentTime, api.ResourceType_RESOURCE_TYPE_VOLUME)
	require.NoError(t, err, "Failed to enumerate results")
	require.Equal(t, 3, len(enAlerts), "Enumerated incorrect number of alert")

	err = kva.Erase(api.ResourceType_RESOURCE_TYPE_VOLUME, raiseAlert1.Id)
	err = kva.Erase(api.ResourceType_RESOURCE_TYPE_VOLUME, raiseAlert2.Id)
	err = kva.Erase(api.ResourceType_RESOURCE_TYPE_VOLUME, raiseAlert3.Id)
	err = kva.Erase(api.ResourceType_RESOURCE_TYPE_NODE, raiseAlert4.Id)
	err = kva.Erase(api.ResourceType_RESOURCE_TYPE_VOLUME, fakeAlertId)
}

func testAlertWatcher(alert *api.Alert, action api.AlertActionType, prefix string, key string) error {
	// A dummy callback function
	// Setting the global variables so that we can check them in our unit tests
	isWatcherCalled = 1
	if action != api.AlertActionType_ALERT_ACTION_TYPE_DELETE {
		watcherAlert = *alert
	} else {
		watcherAlert = api.Alert{}
	}
	watcherAction = action
	watcherPrefix = prefix
	watcherKey = key
	return nil

}

func watch(t *testing.T) {
	isWatcherCalled = 0

	err := kva.Watch(clusterName, testAlertWatcher)
	require.NoError(t, err, "Failed to subscribe a watch function")

	raiseAlert1 := api.Alert{
		Resource: api.ResourceType_RESOURCE_TYPE_CLUSTER,
		Severity: api.SeverityType_SEVERITY_TYPE_NOTIFY,
	}
	err = kva.Raise(&raiseAlert1)

	// Sleep for sometime so that we pass on some previous watch callbacks
	time.Sleep(time.Millisecond * 100)

	require.Equal(t, 1, isWatcherCalled, "Callback function not called")
	require.Equal(t, api.AlertActionType_ALERT_ACTION_TYPE_CREATE, watcherAction, "action mismatch for create")
	require.Equal(t, raiseAlert1.Id, watcherAlert.Id, "alert id mismatch")
	require.Equal(t, "alert/cluster/"+strconv.FormatInt(raiseAlert1.Id, 10), watcherKey, "key mismatch")

	err = kva.Clear(api.ResourceType_RESOURCE_TYPE_CLUSTER, raiseAlert1.Id, 0)

	// Sleep for sometime so that we pass on some previous watch callbacks
	time.Sleep(time.Millisecond * 100)

	require.Equal(t, api.AlertActionType_ALERT_ACTION_TYPE_UPDATE, watcherAction, "action mismatch for update")
	require.Equal(t, "alert/cluster/"+strconv.FormatInt(raiseAlert1.Id, 10), watcherKey, "key mismatch")

	err = kva.Erase(api.ResourceType_RESOURCE_TYPE_CLUSTER, raiseAlert1.Id)

	// Sleep for sometime so that we pass on some previous watch callbacks
	time.Sleep(time.Millisecond * 100)

	require.Equal(t, api.AlertActionType_ALERT_ACTION_TYPE_DELETE, watcherAction, "action mismatch for delete")
	require.Equal(t, "alert/cluster/"+strconv.FormatInt(raiseAlert1.Id, 10), watcherKey, "key mismatch")

	// Watch on a new clusterID
	isWatcherCalled = 0
	err = kva.Watch(newClusterName, testAlertWatcher)

	// Create a new alert instance for raising an alert in this new cluster id
	kvaNew, err := New("alert_kvdb_test", newClusterName, kvdb.Instance())

	raiseAlertNew := api.Alert{
		Resource: api.ResourceType_RESOURCE_TYPE_NODE,
		Severity: api.SeverityType_SEVERITY_TYPE_ALARM,
	}
	err = kvaNew.Raise(&raiseAlertNew)
	// Sleep for sometime so that we pass on some previous watch callbacks
	time.Sleep(time.Millisecond * 100)

	require.Equal(t, 1, isWatcherCalled, "Callback function not called")
	require.Equal(t, api.AlertActionType_ALERT_ACTION_TYPE_CREATE, watcherAction, "action mismatch for create")
	require.Equal(t, raiseAlertNew.Id, watcherAlert.Id, "alert id mismatch")
	require.Equal(t, "alert/node/"+strconv.FormatInt(raiseAlertNew.Id, 10), watcherKey, "key mismatch")

	err = kva.Erase(api.ResourceType_RESOURCE_TYPE_NODE, raiseAlertNew.Id)
}
