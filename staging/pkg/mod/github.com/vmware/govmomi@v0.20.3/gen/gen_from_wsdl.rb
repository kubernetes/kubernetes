# Copyright (c) 2014 VMware, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

$:.unshift(File.expand_path(File.dirname(__FILE__)))

require "vim_wsdl"

if !File.directory?(ARGV.first)
  raise "first argument not a directory"
end

target = ARGV[1]
wsdl = WSDL.new(WSDL.read target+".wsdl")
wsdl.validate_assumptions!
wsdl.peek()

ifs = Peek.types.keys.select { |name| Peek.base?(name) }.size()
puts "%d classes, %d interfaces" % [Peek.types.size(), ifs]

File.open(File.join(ARGV.first, "types/enum.go"), "w") do |io|
  io.print WSDL.header("types")

  wsdl.
    types.
    sort_by { |x| x.name }.
    uniq { |x| x.name }.
    select { |t| t.is_enum? }.
    each { |e| e.dump(io); e.dump_init(io) }
end

File.open(File.join(ARGV.first, "types/types.go"), "w") do |io|
  io.print WSDL.header("types")
  if target != "vim"
    io.print <<EOF
import (
        "context"
        "github.com/vmware/govmomi/vim25/types"
)
EOF
  end

  wsdl.
    types.
    sort_by { |x| x.name }.
    uniq { |x| x.name }.
    select { |t| !t.is_enum? }.
    each { |e| e.dump(io); e.dump_init(io) }
end

File.open(File.join(ARGV.first, "types/if.go"), "w") do |io|
  io.print WSDL.header("types")

  Peek.dump_interfaces(io)
end

File.open(File.join(ARGV.first, "methods/methods.go"), "w") do |io|
  io.print WSDL.header("methods")
  if target == "vim"
    target += "25"
  end

  io.print <<EOF
import (
        "context"
        "github.com/vmware/govmomi/#{target}/types"
        "github.com/vmware/govmomi/vim25/soap"
)
EOF

  wsdl.
    operations.
    sort_by { |x| x.name }.
    each { |e| e.dump(io) }
end

exit(0)
