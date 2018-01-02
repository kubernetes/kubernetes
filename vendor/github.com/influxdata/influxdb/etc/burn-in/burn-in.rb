require "influxdb"
require "colorize"
require "benchmark"

require_relative "log"
require_relative "random_gaussian"

BATCH_SIZE = 10_000

Log.info "Starting burn-in suite"
master = InfluxDB::Client.new
master.delete_database("burn-in") rescue nil
master.create_database("burn-in")
master.create_database_user("burn-in", "user", "pass")

master.database = "burn-in"
# master.query "select * from test1 into test2;"
# master.query "select count(value) from test1 group by time(1m) into test2;"

influxdb = InfluxDB::Client.new "burn-in", username: "user", password: "pass"

Log.success "Connected to server #{influxdb.host}:#{influxdb.port}"

Log.log "Creating RandomGaussian(500, 25)"
gaussian = RandomGaussian.new(500, 25)
point_count = 0

while true
  Log.log "Generating 10,000 points.."
  points = []
  BATCH_SIZE.times do |n|
    points << {value: gaussian.rand.to_i.abs}
  end
  point_count += points.length

  Log.info "Sending points to server.."
  begin
    st = Time.now
    foo = influxdb.write_point("test1", points)
    et = Time.now
    Log.log foo.inspect
    Log.log "#{et-st} seconds elapsed"
    Log.success "Write successful."
  rescue => e
    Log.failure "Write failed:"
    Log.log e
  end
  sleep 0.5

  Log.info "Checking regular points"
  st = Time.now
  response = influxdb.query("select count(value) from test1;")
  et = Time.now

  Log.log "#{et-st} seconds elapsed"

  response_count = response["test1"].first["count"]
  if point_count == response_count
    Log.success "Point counts match: #{point_count} == #{response_count}"
  else
    Log.failure "Point counts don't match: #{point_count} != #{response_count}"
  end

  # Log.info "Checking continuous query points for test2"
  # st = Time.now
  # response = influxdb.query("select count(value) from test2;")
  # et = Time.now

  # Log.log "#{et-st} seconds elapsed"

  # response_count = response["test2"].first["count"]
  # if point_count == response_count
    # Log.success "Point counts match: #{point_count} == #{response_count}"
  # else
    # Log.failure "Point counts don't match: #{point_count} != #{response_count}"
  # end
end


