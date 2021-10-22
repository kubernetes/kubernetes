/*
 *
 * Copyright 2019 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package main

import (
	"context"
	"encoding/gob"
	"fmt"
	"os"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/grpclog"
	ppb "google.golang.org/grpc/profiling/proto"
)

func setEnabled(ctx context.Context, c ppb.ProfilingClient, enabled bool) error {
	_, err := c.Enable(ctx, &ppb.EnableRequest{Enabled: enabled})
	if err != nil {
		grpclog.Infof("error calling Enable: %v\n", err)
		return err
	}

	grpclog.Infof("successfully set enabled = %v", enabled)
	return nil
}

func retrieveSnapshot(ctx context.Context, c ppb.ProfilingClient, f string) error {
	grpclog.Infof("getting stream stats")
	resp, err := c.GetStreamStats(ctx, &ppb.GetStreamStatsRequest{})
	if err != nil {
		grpclog.Errorf("error calling GetStreamStats: %v\n", err)
		return err
	}
	s := &snapshot{StreamStats: resp.StreamStats}

	grpclog.Infof("creating snapshot file %s", f)
	file, err := os.Create(f)
	if err != nil {
		grpclog.Errorf("cannot create %s: %v", f, err)
		return err
	}
	defer file.Close()

	grpclog.Infof("encoding data and writing to snapshot file %s", f)
	encoder := gob.NewEncoder(file)
	err = encoder.Encode(s)
	if err != nil {
		grpclog.Infof("error encoding: %v", err)
		return err
	}

	grpclog.Infof("successfully wrote profiling snapshot to %s", f)
	return nil
}

func remoteCommand() error {
	ctx := context.Background()
	if *flagTimeout > 0 {
		ctx, _ = context.WithTimeout(context.Background(), time.Duration(*flagTimeout)*time.Second)
	}

	grpclog.Infof("dialing %s", *flagAddress)
	cc, err := grpc.Dial(*flagAddress, grpc.WithInsecure())
	if err != nil {
		grpclog.Errorf("cannot dial %s: %v", *flagAddress, err)
		return err
	}
	defer cc.Close()

	c := ppb.NewProfilingClient(cc)

	if *flagEnableProfiling || *flagDisableProfiling {
		return setEnabled(ctx, c, *flagEnableProfiling)
	} else if *flagRetrieveSnapshot {
		return retrieveSnapshot(ctx, c, *flagSnapshot)
	} else {
		return fmt.Errorf("what should I do with the remote target?")
	}
}
