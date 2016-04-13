require 'yard'
require 'yard-go'

module GoLinksHelper
  def signature(obj, link = true, show_extras = true, full_attr_name = true)
    case obj
    when YARDGo::CodeObjects::FuncObject
      if link && obj.has_tag?(:service_operation)
        ret = signature_types(obj, !link)
        args = obj.parameters.map {|m| m[0].split(/\s+/).last }.join(", ")
        line = "<strong>#{obj.name}</strong>(#{args}) #{ret}"
        return link ? linkify(obj, line) : line
      end
    end

    super(obj, link, show_extras, full_attr_name)
  end

  def html_syntax_highlight(source, type = nil)
    src = super(source, type || :go)
    object.has_tag?(:service_operation) ? link_types(src) : src
  end
end

YARD::Templates::Helpers::HtmlHelper.send(:prepend, GoLinksHelper)
YARD::Templates::Engine.register_template_path(File.dirname(__FILE__) + '/templates')

YARD::Parser::SourceParser.after_parse_list do
  YARD::Registry.all(:struct).each do |obj|
    if obj.file =~ /\/?service\/(.+?)\/(service|api)\.go$/
      obj.add_tag YARD::Tags::Tag.new(:service, $1)
      obj.groups = ["Constructor Functions", "Service Operations", "Request Methods", "Pagination Methods"]
    end
  end

  YARD::Registry.all(:method).each do |obj|
    if obj.file =~ /service\/.+?\/api\.go$/ && obj.scope == :instance
      if obj.name.to_s =~ /Pages$/
        obj.group = "Pagination Methods"
        opname = obj.name.to_s.sub(/Pages$/, '')
        obj.docstring = <<-eof
#{obj.name} iterates over the pages of a {#{opname} #{opname}()} operation, calling the `fn`
function callback with the response data in each page. To stop iterating, return `false` from
the function callback.

@note This operation can generate multiple requests to a service.
@example Iterating over at most 3 pages of a #{opname} operation
  pageNum := 0
  err := client.#{obj.name}(params, func(page *#{obj.parent.parent.name}.#{obj.parameters[1][0].split("*").last}, lastPage bool) bool {
    pageNum++
    fmt.Println(page)
    return pageNum <= 3
  })
@see #{opname}
eof
        obj.add_tag YARD::Tags::Tag.new(:paginator, '')
      elsif obj.name.to_s =~ /Request$/
        obj.group = "Request Methods"
        obj.signature = obj.name.to_s
        obj.parameters = []
        opname = obj.name.to_s.sub(/Request$/, '')
        obj.docstring = <<-eof
#{obj.name} generates a {aws/request.Request} object representing the client request for
the {#{opname} #{opname}()} operation. The `output` return value can be used to capture
response data after {aws/request.Request.Send Request.Send()} is called.

Creating a request object using this method should be used when you want to inject
custom logic into the request lifecycle using a custom handler, or if you want to
access properties on the request object before or after sending the request. If
you just want the service response, call the {#{opname} service operation method}
directly instead.

@note You must call the {aws/request.Request.Send Send()} method on the returned
  request object in order to execute the request.
@example Sending a request using the #{obj.name}() method
  req, resp := client.#{obj.name}(params)
  err := req.Send()

  if err == nil { // resp is now filled
    fmt.Println(resp)
  }
eof
        obj.add_tag YARD::Tags::Tag.new(:request_method, '')
      else
        obj.group = "Service Operations"
        obj.add_tag YARD::Tags::Tag.new(:service_operation, '')
        if ex = obj.tag(:example)
          ex.name = "Calling the #{obj.name} operation"
        end
      end
    end
  end

  apply_docs
end

def apply_docs
  svc_pkg = YARD::Registry.at('service')
  return if svc_pkg.nil?

  pkgs = svc_pkg.children.select {|t| t.type == :package }
  pkgs.each do |pkg|
    svc = pkg.children.find {|t| t.has_tag?(:service) }
    ctor = P(svc, ".New")
    svc_name = ctor.source[/ServiceName:\s*"(.+?)",/, 1]
    api_ver = ctor.source[/APIVersion:\s*"(.+?)",/, 1]
    log.progress "Parsing service documentation for #{svc_name} (#{api_ver})"
    file = Dir.glob("models/apis/#{svc_name}/#{api_ver}/docs-2.json").sort.last
    next if file.nil?

    next if svc.nil?
    exmeth = svc.children.find {|s| s.has_tag?(:service_operation) }
    pkg.docstring += <<-eof

@example Sending a request using the {#{svc.name}} client
  client := #{pkg.name}.New(nil)
  params := &#{pkg.name}.#{exmeth.parameters.first[0].split("*").last}{...}
  resp, err := client.#{exmeth.name}(params)
@see #{svc.name}
@version #{api_ver}
eof

    ctor.docstring += <<-eof

@example Constructing a client using default configuration
  client := #{pkg.name}.New(nil)

@example Constructing a client with custom configuration
  config := aws.NewConfig().WithRegion("us-west-2")
  client := #{pkg.name}.New(config)
eof

    json = JSON.parse(File.read(file))
    if svc
      apply_doc(svc, json["service"])
    end

    json["operations"].each do |op, doc|
      if doc && obj = svc.children.find {|t| t.name.to_s.downcase == op.downcase }
        apply_doc(obj, doc)
      end
    end

    json["shapes"].each do |shape, data|
      shape = shape_name(shape)
      if obj = pkg.children.find {|t| t.name.to_s.downcase == shape.downcase }
        apply_doc(obj, data["base"])
      end

      data["refs"].each do |refname, doc|
        refshape, member = *refname.split("$")
        refshape = shape_name(refshape)
        if refobj = pkg.children.find {|t| t.name.to_s.downcase == refshape.downcase }
          if m = refobj.children.find {|t| t.name.to_s.downcase == member.downcase }
            apply_doc(m, doc || data["base"])
          end
        end
      end if data["refs"]
    end
  end
end

def apply_doc(obj, doc)
  tags = obj.docstring.tags || []
  obj.docstring = clean_docstring(doc)
  tags.each {|t| obj.docstring.add_tag(t) }
end

def shape_name(shape)
  shape.sub(/Request$/, "Input").sub(/Response$/, "Output")
end

def clean_docstring(docs)
  return nil unless docs
  docs = docs.gsub(/<!--.*?-->/m, '')
  docs = docs.gsub(/<fullname>.+?<\/fullname?>/m, '')
  docs = docs.gsub(/<examples?>.+?<\/examples?>/m, '')
  docs = docs.gsub(/<note>\s*<\/note>/m, '')
  docs = docs.gsub(/<a>(.+?)<\/a>/, '\1')
  docs = docs.gsub(/<note>(.+?)<\/note>/m) do
    text = $1.gsub(/<\/?p>/, '')
    "<div class=\"note\"><strong>Note:</strong> #{text}</div>"
  end
  docs = docs.gsub(/\{(.+?)\}/, '`{\1}`')
  docs = docs.gsub(/\s+/, ' ').strip
  docs == '' ? nil : docs
end
