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

require "test/unit"

def read(file)
  File.open(file)
end

class Prop
  def initialize(vmodl, data)
    @vmodl = vmodl
    @data = data
  end

  def slice?
    @data["is-array"]
  end

  def optional?
    @data["is-optional"]
  end

  def name
    @data["name"]
  end

  def var_field
    n = name
    n[0].capitalize + n[1..-1]
  end

  def var_type_prefix(base=false)
    if slice?
      "[]"
    else
      if optional? && !base
        "*"
      else
        ""
      end
    end
  end

  def var_type
    type = @data["wsdl_type"]
    if @vmodl.managed_hash.has_key?(type)
      type = "ManagedObjectReference"
    end

    # Fix up type from vmodl
    case type
    when "TypeName", "MethodName"
      type = "xsd:string"
    when "ManagedObject"
      type = "ManagedObjectReference"
    when "xsd:anyType"
      type = "AnyType"
    end

    if type =~ /^xsd:(.*)$/
      type = $1
      case type
      when "string"
      when "int"
        type = "int32"
      when "boolean"
        type ="bool"
      when "long"
        type ="int64"
      when "dateTime"
        type ="time.Time"
      when "byte"
      when "double"
        type ="float64"
      when "float"
        type ="float32"
      when "short"
        type ="int16"
      when "base64Binary"
        type ="[]byte"
      else
        raise "unknown type: %s" % type
      end
    else
      if Peek.base?(type)
        type = "Base" + type
        base = true
      end
      type = "types." + type
    end

    var_type_prefix(base) + type
  end

  def var_tag
    "mo:\"%s\"" % name
  end

  def dump(io)
    io.print "%s %s `%s`\n" % [var_field, var_type, var_tag]
  end
end

class Managed
  def initialize(vmodl, name, data)
    @vmodl = vmodl
    @name = name
    @data = data
  end

  def name
    @name
  end

  def props
    @data["props"].map do |p|
      Prop.new(@vmodl, p)
    end
  end

  def dump(io)
    include_ref_getter = false
    include_ent_getter = false

    io.print "type %s struct {\n" % name

    case @data["wsdl_base"]
    when nil, "ManagedObject", "View"
      include_ref_getter = true
      io.print "Self types.ManagedObjectReference\n\n"
    else
      io.print "%s\n\n" % @data["wsdl_base"]
      if @data["wsdl_base"] == "ManagedEntity"
        include_ent_getter = true
      end
    end

    seen = {}
    props.each do |p|
      next if seen[p.name]
      p.dump(io)
      seen[p.name] = true
    end
    io.print "}\n\n"

    if include_ref_getter
      io.print "func (m %s) Reference() types.ManagedObjectReference {\n" % [name]
      io.print "return m.Self\n"
      io.print "}\n\n"
    end

    if include_ent_getter
      io.print "func (m *%s) Entity() *ManagedEntity {\n" % [name]
      io.print "return &m.ManagedEntity\n"
      io.print "}\n\n"
    end
  end

  def dump_init(io)
    io.print "func init() {\n"
    io.print "t[\"%s\"] = reflect.TypeOf((*%s)(nil)).Elem()\n" % [name, name]
    io.print "}\n\n"
  end
end

class Vmodl
  def initialize(data)
    @data = Marshal.load(data)
  end

  def managed_hash
    @managed_hash ||= begin
                        h = {}
                        managed.each do |m|
                          h[m.name] = m
                        end
                        h
                      end
  end

  def managed
    @data.map do |k,v|
      next if !v.is_a?(Hash)
      next if v["kind"] != "managed"
      # rbvmomi/vmodl.db includes pbm mo's, but we don't need the types as they have no properties
      next if k =~ /^pbm/i
      # internal/types.go already includes these
      next if ["InternalDynamicTypeManager", "ReflectManagedMethodExecuter"].include?(k)
      Managed.new(self, k, v)
    end.compact
  end
end

if !File.directory?(ARGV.first)
  raise "first argument not a directory"
end

wsdl = WSDL.new(WSDL.read ARGV[1]+".wsdl")
wsdl.validate_assumptions!
wsdl.peek()

vmodl = Vmodl.new(read ARGV[2] || "./rbvmomi/vmodl.db")

File.open(File.join(ARGV.first, "mo/mo.go"), "w") do |io|
  io.print WSDL.header("mo")
  io.print <<EOF
import (
        "github.com/vmware/govmomi/vim25/types"
)
EOF

  vmodl.
    managed.
    sort_by { |m| m.name }.
    each { |m| m.dump(io); m.dump_init(io); }
end

exit(0)
