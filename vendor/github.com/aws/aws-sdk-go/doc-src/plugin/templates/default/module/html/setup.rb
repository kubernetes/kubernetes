def init
  super
  sections.place(:client, [:item_summary]).before(:constant_summary)
end

def client
  @client = object.children.find {|c| c.has_tag?(:service) }
  erb(:client) if @client
end
