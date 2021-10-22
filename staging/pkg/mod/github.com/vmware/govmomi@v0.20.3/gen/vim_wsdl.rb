# Copyright (c) 2014-2018 VMware, Inc. All Rights Reserved.
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

require "nokogiri"
require "test/unit"

$namespaces = %w(vim25)

def valid_ns?(t)
  $namespaces.include?(t)
end

def ucfirst(v)
  x = "ArrayOf"
  if v.start_with?(x)
    # example: ArrayOfvslmInfrastructureObjectPolicy -> ArrayOfVslm...
    return x + ucfirst(v[x.length..-1])
  end

  # example: vslmInfrastructureObjectPolicy -. VslmInfrastructureObjectPolicy
  v[0].capitalize + v[1..-1]
end

def init_type(io, name, kind)
  t = "reflect.TypeOf((*#{ucfirst kind})(nil)).Elem()"

  io.print "func init() {\n"

  if $target == "vim25"
    io.print "t[\"#{name}\"] = #{t}\n"
  else
    unless name.start_with? "Base"
      name = "#{$target}:#{name}"
    end
    io.print "types.Add(\"#{name}\", #{t})\n"
  end

  io.print "}\n\n"
end

class Peek
  class Type
    attr_accessor :parent, :children, :klass

    def initialize(name)
      @name = name
      @children = []
    end

    def base?
      # VrpResourceAllocationInfo is removed in 6.7, so base will no longer generated
      return false if @name == "ResourceAllocationInfo"

      return !children.empty?
    end
  end

  @@types = {}
  @@refs = {}
  @@enums = {}

  def self.types
    return @@types
  end

  def self.refs
    return @@refs
  end

  def self.enums
    return @@enums
  end

  def self.ref(type)
    refs[type] = true
  end

  def self.enum(type)
    enums[type] = true
  end

  def self.enum?(type)
    enums[type]
  end

  def self.register(name)
    raise unless name
    types[name] ||= Type.new(name)
  end

  def self.base?(name)
    return unless c = types[name]
    c.base?
  end

  def self.dump_interfaces(io)
    types.keys.sort.each do |name|
      next unless base?(name)
      klass = types[name].klass
      klass.dump_interface(io, name) if klass
    end
  end
end

class EnumValue
  def initialize(type, value)
    @type = type
    @value = value
  end

  def type_name
    ucfirst(@type.name)
  end

  def var_name
    n = ucfirst(@type.name)
    v = var_value
    if v == ""
      n += "Null"
    else
      n += ucfirst(v)
    end

    return n
  end

  def var_value
    @value
  end

  def dump(io)
    io.print "%s = %s(\"%s\")\n" % [var_name, type_name, var_value]
  end
end

class Simple
  include Test::Unit::Assertions

  attr_accessor :name, :type

  def initialize(node)
    @node = node
  end

  def name
    @name || @node["name"]
  end

  def type
    @type || @node["type"]
  end

  def is_enum?
    false
  end

  def dump_init(io)
    # noop
  end

  def var_name
    n = self.name
    n = n[1..-1] if n[0] == "_" # Strip leading _
    n = ucfirst(n)
    return n
  end

  def ns(t = self.type)
    t.split(":", 2)[0]
  end

  def vim_type?
    valid_ns? ns
  end

  def vim_type(t = self.type)
    ns, kind = t.split(":", 2)
    if ! valid_ns? ns
        raise
    end
    ucfirst(kind)
  end

  def base_type?
    vim_type? && Peek.base?(vim_type)
  end

  def enum_type?
    vim_type? && Peek.enum?(vim_type)
  end

  def any_type?
    self.type == "xsd:anyType"
  end

  def pointer_type?
    ["UnitNumber"].include?(var_name) or
      optional? && ["OwnerId", "GroupId", "MaxWaitSeconds", "Reservation", "Limit", "OverheadLimit"].include?(var_name)
  end

  def var_type
    t = self.type
    prefix = ""

    if slice?
      prefix += "[]"
      if ["AffinitySet"].include?(var_name)
        self.need_omitempty = false
      end
    end

    if t =~ /^xsd:(.*)$/
      t = $1
      case t
      when "string"
      when "int"
        if pointer_type?
          prefix += "*"
          self.need_omitempty = false
        end
        t = "int32"
      when "boolean"
        t = "bool"
        if !slice? && optional?
          prefix += "*"
          self.need_omitempty = false
        end
      when "long"
        if pointer_type?
          prefix += "*"
          self.need_omitempty = false
        end
        t = "int64"
      when "dateTime"
        t = "time.Time"
        if !slice? && optional?
          prefix += "*"
          self.need_omitempty = false
        end
      when "anyType"
        pkg = ""
        if $target != "vim25"
          pkg = "types."
        end
        t = "#{pkg}AnyType"
        if ["Value", "Val"].include?(var_name)
          self.need_omitempty = false
        end
      when "byte"
      when "double"
        t = "float64"
      when "float"
        t = "float32"
      when "short"
        t = "int16"
      when "base64Binary"
        t = "[]byte"
      when "anyURI"
        t = "string"
      else
        raise "unknown type: %s" % t
      end
    else
      pkg = ""
      if $target != self.ns
        pkg = "types."
      end

      t = vim_type

      if base_type?
        prefix += "#{pkg}Base"
      else
        t = pkg + t
        prefix += "*" if !slice? && !enum_type? && optional?
      end
    end

    prefix + t
  end

  def slice?
    test_attr("maxOccurs", "unbounded")
  end

  def optional?
    test_attr("minOccurs", "0")
  end

  def need_omitempty=(v)
    @need_omitempty = v
  end

  def need_omitempty?
    var_type # HACK: trigger setting need_omitempty if necessary
    if @need_omitempty.nil?
      @need_omitempty = optional?
    else
      @need_omitempty
    end
  end

  def need_typeattr?
    base_type? || any_type?
  end

  protected

  def test_attr(attr, expected)
    actual = @node.attr(attr)
    if actual != nil
      case actual
      when expected
        true
      else
        raise "%s=%s" % [value, type.attr(value)]
      end
    else
      false
    end
  end
end

class Element < Simple
  def initialize(node)
    super(node)
  end

  def has_type?
    !@node["type"].nil?
  end

  def child
    cs = @node.element_children
    assert_equal 1, cs.length
    assert_equal "complexType", cs.first.name

    t = ComplexType.new(cs.first)
    t.name = self.name
    t
  end

  def dump(io)
    if has_type?
      io.print "type %s %s\n\n" % [ucfirst(name), var_type]
    else
      child.dump(io)
    end
  end

  def dump_init(io)
    if has_type?
      init_type io, name, name
    end
  end

  def dump_field(io)
    tag = name
    tag += ",omitempty" if need_omitempty?
    tag += ",typeattr" if need_typeattr?
    io.print "%s %s `xml:\"%s\"`\n" % [var_name, var_type, tag]
  end

  def peek(type=nil)
    if has_type?
      return if self.type =~ /^xsd:/

      Peek.ref(vim_type)
    else
      child.peek()
    end
  end
end

class Attribute < Simple
  def dump_field(io)
    tag = name
    tag += ",omitempty" if need_omitempty?
    tag += ",attr"
    io.print "%s %s `xml:\"%s\"`\n" % [var_name, var_type, tag]
  end
end

class SimpleType < Simple
  def is_enum?
    true
  end

  def dump(io)
    enums = @node.xpath(".//xsd:enumeration").map do |n|
      EnumValue.new(self, n["value"])
    end

    io.print "type %s string\n\n" % ucfirst(name)
    io.print "const (\n"
    enums.each { |e| e.dump(io) }
    io.print ")\n\n"
  end

  def dump_init(io)
    init_type io, name, name
  end

  def peek
    Peek.enum(name)
  end
end

class ComplexType < Simple
  class SimpleContent < Simple
    def dump(io)
      attr = Attribute.new(@node.at_xpath(".//xsd:attribute"))
      attr.dump_field(io)

      # HACK DELUXE(PN)
      extension = @node.at_xpath(".//xsd:extension")
      type = extension["base"].split(":", 2)[1]
      io.print "Value %s `xml:\",chardata\"`\n" % type
    end

    def peek
    end
  end

  class ComplexContent < Simple
    def base
      extension = @node.at_xpath(".//xsd:extension")
      assert_not_nil extension

      base = extension["base"]
      assert_not_nil base

      base
    end

    def dump(io)
      Sequence.new(@node).dump(io, base)
    end

    def dump_interface(io, name)
      Sequence.new(@node).dump_interface(io, name)
    end

    def peek
      Sequence.new(@node).peek(vim_type(base))
    end
  end

  class Sequence < Simple
    def sequence
      sequence = @node.at_xpath(".//xsd:sequence")
      if sequence != nil
        sequence.element_children.map do |n|
          Element.new(n)
        end
      else
        nil
      end
    end

    def dump(io, base = nil)
      return unless elements = sequence
      if base != nil
        kind = vim_type(base)

        pkg = ""
        if $target != ns(base)
          pkg = "types."
        end
        io.print "#{pkg}#{kind}\n\n"
      end

      elements.each do |e|
        e.dump_field(io)
      end
    end

    def dump_interface(io, name)
      method = "Get%s() *%s" % [name, name]
      io.print "func (b *%s) %s { return b }\n" % [name, method]
      io.print "type Base%s interface {\n" % name
      io.print "%s\n" % method
      io.print "}\n\n"
      init_type io, "Base#{name}", name
    end

    def peek(base = nil)
      return unless elements = sequence
      name = @node.attr("name")
      return unless name

      elements.each do |e|
        e.peek(name)
      end

      c = Peek.register(name)
      if base
        c.parent = base
        Peek.register(c.parent).children << name
      end
    end
  end

  def klass
    @klass ||= begin
                 cs = @node.element_children
                 if !cs.empty?
                   assert_equal 1, cs.length

                   case cs.first.name
                   when "simpleContent"
                     SimpleContent.new(@node)
                   when "complexContent"
                     ComplexContent.new(@node)
                   when "sequence"
                     Sequence.new(@node)
                   else
                     raise "don't know what to do for element: %s..." % cs.first.name
                   end
                 end
               end
  end

  def dump_init(io)
    init_type io, name, name
  end

  def dump(io)
    io.print "type %s struct {\n" % ucfirst(name)
    klass.dump(io) if klass
    io.print "}\n\n"
  end

  def peek
    Peek.register(name).klass = klass
    klass.peek if klass
  end
end

class Schema
  include Test::Unit::Assertions

  attr_accessor :namespace

  def initialize(xml)
    @xml = Nokogiri::XML.parse(xml)
    @namespace = @xml.root.attr("targetNamespace").split(":", 2)[1]
    @xml
  end

  # We have some assumptions about structure, make sure they hold.
  def validate_assumptions!
    # Every enumeration is part of a restriction
    @xml.xpath(".//xsd:enumeration").each do |n|
      assert_equal "restriction", n.parent.name
    end

    # See type == enum
    @xml.xpath(".//xsd:restriction").each do |n|
      # Every restriction has type xsd:string (it's an enum)
      assert_equal "xsd:string", n["base"]

      # Every restriction is part of a simpleType
      assert_equal "simpleType", n.parent.name

      # Every restriction is alone
      assert_equal 1, n.parent.element_children.size
    end

    # See type == complex_content
    @xml.xpath(".//xsd:complexContent").each do |n|
      # complexContent is child of complexType
      assert_equal "complexType", n.parent.name

    end

    # See type == complex_type
    @xml.xpath(".//xsd:complexType").each do |n|
      cc = n.element_children

      # OK to have an empty complexType
      next if cc.size == 0

      # Require 1 element otherwise
      assert_equal 1, cc.size

      case cc.first.name
      when "complexContent"
        # complexContent has 1 "extension" element
        cc = cc.first.element_children
        assert_equal 1, cc.size
        assert_equal "extension", cc.first.name

        # extension has 1 "sequence" element
        ec = cc.first.element_children
        assert_equal 1, ec.size
        assert_equal "sequence", ec.first.name

        # sequence has N "element" elements
        sc = ec.first.element_children
        assert sc.all? { |e| e.name == "element" }
      when "simpleContent"
        # simpleContent has 1 "extension" element
        cc = cc.first.element_children
        assert_equal 1, cc.size
        assert_equal "extension", cc.first.name

        # extension has 1 or more "attribute" elements
        ec = cc.first.element_children
        assert_not_equal 0, ec.size
        assert_equal "attribute", ec.first.name
      when "sequence"
        # sequence has N "element" elements
        sc = cc.first.element_children
        assert sc.all? { |e| e.name == "element" }
      else
        raise "unknown element: %s" % cc.first.name
      end
    end

    imports.each do |i|
      i.validate_assumptions!
    end

    includes.each do |i|
      i.validate_assumptions!
    end
  end

  def types
    return to_enum(:types) unless block_given?

    if $target != self.namespace
      return
    end

    imports.each do |i|
      i.types do |t|
        yield t
      end
    end

    includes.each do |i|
      i.types do |t|
        yield t
      end
    end

    @xml.root.children.each do |n|
      case n.class.to_s
      when "Nokogiri::XML::Text"
        next
      when "Nokogiri::XML::Element"
        case n.name
        when "include", "import"
          next
        when "element"
          e = Element.new(n)
          if e.has_type? && e.vim_type?
            if e.ns == $target
              yield e
            end
          else
            yield e
          end
        when "simpleType"
          yield SimpleType.new(n)
        when "complexType"
          yield ComplexType.new(n)
        else
          raise "unknown child: %s" % n.name
        end
      else
        raise "unknown type: %s" % n.class
      end
    end
  end

  def imports
    @imports ||= @xml.root.xpath(".//xmlns:import").map do |n|
      Schema.new(WSDL.read n["schemaLocation"])
    end
  end

  def includes
    @includes ||= @xml.root.xpath(".//xmlns:include").map do |n|
      Schema.new(WSDL.read n["schemaLocation"])
    end
  end
end


class Operation
  include Test::Unit::Assertions

  def initialize(wsdl, operation_node)
    @wsdl = wsdl
    @operation_node = operation_node
  end

  def name
    @operation_node["name"]
  end

  def namespace
    type = @operation_node.at_xpath("./xmlns:input").attr("message")
    keep_ns(type)
  end

  def remove_ns(x)
    ns, x = x.split(":", 2)
    if ! valid_ns? ns
        raise
    end
    x
  end

  def keep_ns(x)
    ns, x = x.split(":", 2)
    if ! valid_ns? ns
        raise
    end
    ns
  end

  def find_type_for(type)
    type = remove_ns(type)

    message = @wsdl.message(type)
    assert_not_nil message

    part = message.at_xpath("./xmlns:part")
    assert_not_nil message

    remove_ns(part["element"])
  end

  def input
    type = @operation_node.at_xpath("./xmlns:input").attr("message")
    find_type_for(type)
  end

  def go_input
    "types." + ucfirst(input)
  end

  def output
    type = @operation_node.at_xpath("./xmlns:output").attr("message")
    find_type_for(type)
  end

  def go_output
    "types." + ucfirst(output)
  end

  def dump(io)
    func = ucfirst(name)
    io.print <<EOS
  type #{func}Body struct{
    Req *#{go_input} `xml:"urn:#{namespace} #{input},omitempty"`
    Res *#{go_output} `xml:"urn:#{namespace} #{output},omitempty"`
    Fault_ *soap.Fault `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
  }

  func (b *#{func}Body) Fault() *soap.Fault { return b.Fault_ }

EOS

    io.print "func %s(ctx context.Context, r soap.RoundTripper, req *%s) (*%s, error) {\n" % [func, go_input, go_output]
    io.print <<EOS
  var reqBody, resBody #{func}Body

  reqBody.Req = req

  if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
    return nil, err
  }

  return resBody.Res, nil
EOS

    io.print "}\n\n"
  end
end

class WSDL
  attr_reader :xml

  PATH = File.expand_path("../sdk", __FILE__)

  def self.read(file)
    File.open(File.join(PATH, file))
  end

  def initialize(xml)
    @xml = Nokogiri::XML.parse(xml)
    $target = @xml.root["targetNamespace"].split(":", 2)[1]

    unless $namespaces.include? $target
      $namespaces.push $target
    end
  end

  def validate_assumptions!
    schemas.each do |s|
      s.validate_assumptions!
    end
  end

  def types(&blk)
    return to_enum(:types) unless block_given?

    schemas.each do |s|
      s.types(&blk)
    end
  end

  def schemas
    @schemas ||= @xml.xpath('.//xmlns:types/xsd:schema').map do |n|
      Schema.new(n.to_xml)
    end
  end

  def operations
    @operations ||= @xml.xpath('.//xmlns:portType/xmlns:operation').map do |o|
      Operation.new(self, o)
    end
  end

  def message(type)
    @messages ||= begin
                    h = {}
                    @xml.xpath('.//xmlns:message').each do |n|
                      h[n.attr("name")] = n
                    end
                    h
                  end

    @messages[type]
  end

  def peek
    types.
      sort_by { |x| x.name }.
      uniq { |x| x.name }.
      each { |e| e.peek() }
  end

  def self.header(name)
    return <<EOF
/*
Copyright (c) 2014-2018 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package #{name}

EOF
  end
end
