# List following and followers from several accounts

users = [
  'PhantomJS'
  'ariyahidayat'
  'detronizator'
  'KDABQt'
  'lfranchi'
  'jonleighton'
  '_jamesmgreene'
  'Vitalliumm'
  ]

follow = (user, callback) ->
  page = require('webpage').create()
  page.open 'http://mobile.twitter.com/' + user, (status) ->
    if status is 'fail'
      console.log user + ': ?'
    else
      data = page.evaluate -> document.querySelector('div.profile td.stat.stat-last div.statnum').innerText;
      console.log user + ': ' + data
    page.close()
    callback.apply()

process = () ->
  if (users.length > 0)
    user = users[0]
    users.splice(0, 1)
    follow(user, process)
  else
    phantom.exit()

process()
