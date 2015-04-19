system = require 'system'

system.stdout.write 'Hello, system.stdout.write!'
system.stdout.writeLine '\nHello, system.stdout.writeLine!'

system.stderr.write 'Hello, system.stderr.write!'
system.stderr.writeLine '\nHello, system.stderr.writeLine!'

system.stdout.writeLine 'system.stdin.readLine(): '
line = system.stdin.readLine()
system.stdout.writeLine JSON.stringify line

# This is essentially a `readAll`
system.stdout.writeLine 'system.stdin.read(5): (ctrl+D to end)'
input = system.stdin.read 5
system.stdout.writeLine JSON.stringify input

phantom.exit 0
