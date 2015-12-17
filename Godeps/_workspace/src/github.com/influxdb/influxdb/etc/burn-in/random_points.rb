require "influxdb"

ONE_WEEK_IN_SECONDS = 7*24*60*60
NUM_POINTS = 10_000
BATCHES = 100

master = InfluxDB::Client.new
master.delete_database("ctx") rescue nil
master.create_database("ctx")

influxdb = InfluxDB::Client.new "ctx"
influxdb.time_precision = "s"

names = ["foo", "bar", "baz", "quu", "qux"]

st = Time.now
BATCHES.times do |m|
  points = []

  puts "Writing #{NUM_POINTS} points, time ##{m}.."
  NUM_POINTS.times do |n|
    timestamp = Time.now.to_i - rand(ONE_WEEK_IN_SECONDS)
    points << {value: names.sample, time: timestamp}
  end

  influxdb.write_point("ct1", points)
end
puts st
puts Time.now
