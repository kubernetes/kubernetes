module Log
  def self.info(msg)
    print Time.now.strftime("%r") + " | "
    puts msg.to_s.colorize(:yellow)
  end

  def self.success(msg)
    print Time.now.strftime("%r") + " | "
    puts msg.to_s.colorize(:green)
  end

  def self.failure(msg)
    print Time.now.strftime("%r") + " | "
    puts msg.to_s.colorize(:red)
  end

  def self.log(msg)
    print Time.now.strftime("%r") + " | "
    puts msg.to_s
  end
end


