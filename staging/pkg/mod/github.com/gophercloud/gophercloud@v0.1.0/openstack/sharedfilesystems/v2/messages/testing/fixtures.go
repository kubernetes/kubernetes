package testing

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func MockDeleteResponse(t *testing.T) {
	th.Mux.HandleFunc("/messages/messageID", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusAccepted)
	})
}

func MockListResponse(t *testing.T) {
	th.Mux.HandleFunc("/messages", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
        {
          "messages": [
            {
              "resource_id": "0d0b883f-95ef-406c-b930-55612ee48a6d",
              "message_level": "ERROR",
              "user_message": "create: Could not find an existing share server or allocate one on the share network provided. You may use a different share network, or verify the network details in the share network and retry your request. If this doesn't work, contact your administrator to troubleshoot issues with your network.",
              "expires_at": "2019-01-06T08:53:38.000000",
              "id": "143a6cc2-1998-44d0-8356-22070b0ebdaa",
              "created_at": "2018-12-07T08:53:38.000000",
              "detail_id": "004",
              "request_id": "req-21767eee-22ca-40a4-b6c0-ae7d35cd434f",
              "project_id": "a5e9d48232dc4aa59a716b5ced963584",
              "resource_type": "SHARE",
              "action_id": "002"
            },
            {
              "resource_id": "4336d74f-3bdc-4f27-9657-c01ec63680bf",
              "message_level": "ERROR",
              "user_message": "create: Could not find an existing share server or allocate one on the share network provided. You may use a different share network, or verify the network details in the share network and retry your request. If this doesn't work, contact your administrator to troubleshoot issues with your network.",
              "expires_at": "2019-01-06T08:53:34.000000",
              "id": "2076373e-13a7-4b84-9e67-15ce8cceaff8",
              "created_at": "2018-12-07T08:53:34.000000",
              "detail_id": "004",
              "request_id": "req-957792ed-f38b-42db-a86a-850f815cbbe9",
              "project_id": "a5e9d48232dc4aa59a716b5ced963584",
              "resource_type": "SHARE",
              "action_id": "002"
            }
          ]
        }`)
	})
}

func MockFilteredListResponse(t *testing.T) {
	th.Mux.HandleFunc("/messages", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
        {
          "messages": [
            {
              "resource_id": "4336d74f-3bdc-4f27-9657-c01ec63680bf",
              "message_level": "ERROR",
              "user_message": "create: Could not find an existing share server or allocate one on the share network provided. You may use a different share network, or verify the network details in the share network and retry your request. If this doesn't work, contact your administrator to troubleshoot issues with your network.",
              "expires_at": "2019-01-06T08:53:34.000000",
              "id": "2076373e-13a7-4b84-9e67-15ce8cceaff8",
              "created_at": "2018-12-07T08:53:34.000000",
              "detail_id": "004",
              "request_id": "req-957792ed-f38b-42db-a86a-850f815cbbe9",
              "project_id": "a5e9d48232dc4aa59a716b5ced963584",
              "resource_type": "SHARE",
              "action_id": "002"
            }
          ]
        }`)
	})
}

func MockGetResponse(t *testing.T) {
	th.Mux.HandleFunc("/messages/2076373e-13a7-4b84-9e67-15ce8cceaff8", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
        {
          "message": {
            "resource_id": "4336d74f-3bdc-4f27-9657-c01ec63680bf",
            "message_level": "ERROR",
            "user_message": "create: Could not find an existing share server or allocate one on the share network provided. You may use a different share network, or verify the network details in the share network and retry your request. If this doesn't work, contact your administrator to troubleshoot issues with your network.",
            "expires_at": "2019-01-06T08:53:34.000000",
            "id": "2076373e-13a7-4b84-9e67-15ce8cceaff8",
            "created_at": "2018-12-07T08:53:34.000000",
            "detail_id": "004",
            "request_id": "req-957792ed-f38b-42db-a86a-850f815cbbe9",
            "project_id": "a5e9d48232dc4aa59a716b5ced963584",
            "resource_type": "SHARE",
            "action_id": "002"
          }
        }`)
	})
}
