#!/usr/bin/env ruby

require 'json'
require 'date'

class Numeric
  def duration
    secs  = self.to_int
    mins  = secs / 60
    hours = mins / 60
    days  = hours / 24

    if days > 0
      "#{days} days and #{hours % 24} hours"
    elsif hours > 0
      "#{hours} hours and #{mins % 60} minutes"
    elsif mins > 0
      "#{mins} minutes and #{secs % 60} seconds"
    elsif secs >= 0
      "#{secs} seconds"
    end
  end
end

pods_json = `kubectl get pods -o json`
pods_parsed = JSON.parse(pods_json)

puts "The Magnificent Aging Plugin."

data = Hash.new
max_name_length = 0
max_age = 0
min_age = 0

pods_parsed['items'].each { |pod|
  name = pod['metadata']['name']
  creation = pod['metadata']['creationTimestamp']

  age = Time.now - DateTime.parse(creation).to_time
  data[name] = age

  if name.length > max_name_length
    max_name_length = name.length
  end
  if age > max_age
    max_age = age
  end
  if age < min_age
    min_age = age
  end
} 

data = data.sort_by{ |name, age| age }

if data.length > 0
  puts ""
  data.each { |name, age|
    output = ""
    output += name.rjust(max_name_length, ' ') + ": "
    bar_size = (age*80/max_age).ceil 
    bar_size.times{ output += "▒" }
    output += " " + age.duration
    puts output
    puts ""
  }
else
  puts "No pods"
end

