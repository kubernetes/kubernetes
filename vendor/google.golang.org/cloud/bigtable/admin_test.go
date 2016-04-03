package bigtable

import (
	"reflect"
	"sort"
	"testing"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/cloud"
	"google.golang.org/cloud/bigtable/bttest"
	"google.golang.org/grpc"
)

func TestAdminIntegration(t *testing.T) {
	srv, err := bttest.NewServer()
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	t.Logf("bttest.Server running on %s", srv.Addr)

	ctx, _ := context.WithTimeout(context.Background(), 2*time.Second)

	conn, err := grpc.Dial(srv.Addr, grpc.WithInsecure())
	if err != nil {
		t.Fatalf("grpc.Dial: %v", err)
	}

	adminClient, err := NewAdminClient(ctx, "proj", "zone", "cluster", cloud.WithBaseGRPC(conn))
	if err != nil {
		t.Fatalf("NewAdminClient: %v", err)
	}
	defer adminClient.Close()

	list := func() []string {
		tbls, err := adminClient.Tables(ctx)
		if err != nil {
			t.Fatalf("Fetching list of tables: %v", err)
		}
		sort.Strings(tbls)
		return tbls
	}
	if err := adminClient.CreateTable(ctx, "mytable"); err != nil {
		t.Fatalf("Creating table: %v", err)
	}
	if err := adminClient.CreateTable(ctx, "myothertable"); err != nil {
		t.Fatalf("Creating table: %v", err)
	}
	if got, want := list(), []string{"myothertable", "mytable"}; !reflect.DeepEqual(got, want) {
		t.Errorf("adminClient.Tables returned %#v, want %#v", got, want)
	}
	if err := adminClient.DeleteTable(ctx, "myothertable"); err != nil {
		t.Fatalf("Deleting table: %v", err)
	}
	if got, want := list(), []string{"mytable"}; !reflect.DeepEqual(got, want) {
		t.Errorf("adminClient.Tables returned %#v, want %#v", got, want)
	}
}
