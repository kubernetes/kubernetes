def init
  super
  sections.place(:request_methods, :paginators).after(:method_summary)
end

def groups(list, type = "Method")
  super(list.reject {|o| o.has_tag?(:paginator) || o.has_tag?(:request_method) }, type)
end

def paginators
  @items = object.children.select {|o| o.has_tag?(:paginator) }
  return if @items.size == 0
  erb(:paginators)
end

def request_methods
  @items = object.children.select {|o| o.has_tag?(:request_method) }
  return if @items.size == 0
  erb(:request_methods)
end
