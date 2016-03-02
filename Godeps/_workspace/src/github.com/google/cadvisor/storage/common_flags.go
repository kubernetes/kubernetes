// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package storage

import (
	"flag"
	"time"
)

var ArgDbUsername = flag.String("storage_driver_user", "root", "database username")
var ArgDbPassword = flag.String("storage_driver_password", "root", "database password")
var ArgDbHost = flag.String("storage_driver_host", "localhost:8086", "database host:port")
var ArgDbName = flag.String("storage_driver_db", "cadvisor", "database name")
var ArgDbTable = flag.String("storage_driver_table", "stats", "table name")
var ArgDbIsSecure = flag.Bool("storage_driver_secure", false, "use secure connection with database")
var ArgDbBufferDuration = flag.Duration("storage_driver_buffer_duration", 60*time.Second, "Writes in the storage driver will be buffered for this duration, and committed to the non memory backends as a single transaction")
