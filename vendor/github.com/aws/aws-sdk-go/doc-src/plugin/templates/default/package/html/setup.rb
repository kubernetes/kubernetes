def type_summary
  @items = object.children.
    select {|c| c.type == :bare_struct || c.type == :struct || c.type == :enum }.
    reject {|c| c.has_tag?(:service) }.
    sort_by {|c| c.name.to_s }
  @name = "Type"
  erb :list_summary
end
