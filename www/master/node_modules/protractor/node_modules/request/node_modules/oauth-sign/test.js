var hmacsign = require('./index').hmacsign
  , assert = require('assert')
  , qs = require('querystring')
  ;

// Tests from Twitter documentation https://dev.twitter.com/docs/auth/oauth

var reqsign = hmacsign('POST', 'https://api.twitter.com/oauth/request_token', 
  { oauth_callback: 'http://localhost:3005/the_dance/process_callback?service_provider_id=11'
  , oauth_consumer_key: 'GDdmIQH6jhtmLUypg82g'
  , oauth_nonce: 'QP70eNmVz8jvdPevU3oJD2AfF7R7odC2XJcn4XlZJqk'
  , oauth_signature_method: 'HMAC-SHA1'
  , oauth_timestamp: '1272323042'
  , oauth_version: '1.0'
  }, "MCD8BKwGdgPHvAuvgvz4EQpqDAtx89grbuNMRd7Eh98")

console.log(reqsign)
console.log('8wUi7m5HFQy76nowoCThusfgB+Q=')
assert.equal(reqsign, '8wUi7m5HFQy76nowoCThusfgB+Q=')

var accsign = hmacsign('POST', 'https://api.twitter.com/oauth/access_token',
  { oauth_consumer_key: 'GDdmIQH6jhtmLUypg82g'
  , oauth_nonce: '9zWH6qe0qG7Lc1telCn7FhUbLyVdjEaL3MO5uHxn8'
  , oauth_signature_method: 'HMAC-SHA1'
  , oauth_token: '8ldIZyxQeVrFZXFOZH5tAwj6vzJYuLQpl0WUEYtWc'
  , oauth_timestamp: '1272323047'
  , oauth_verifier: 'pDNg57prOHapMbhv25RNf75lVRd6JDsni1AJJIDYoTY'
  , oauth_version: '1.0'
  }, "MCD8BKwGdgPHvAuvgvz4EQpqDAtx89grbuNMRd7Eh98", "x6qpRnlEmW9JbQn4PQVVeVG8ZLPEx6A0TOebgwcuA")
  
console.log(accsign)
console.log('PUw/dHA4fnlJYM6RhXk5IU/0fCc=')
assert.equal(accsign, 'PUw/dHA4fnlJYM6RhXk5IU/0fCc=')

var upsign = hmacsign('POST', 'http://api.twitter.com/1/statuses/update.json', 
  { oauth_consumer_key: "GDdmIQH6jhtmLUypg82g"
  , oauth_nonce: "oElnnMTQIZvqvlfXM56aBLAf5noGD0AQR3Fmi7Q6Y"
  , oauth_signature_method: "HMAC-SHA1"
  , oauth_token: "819797-Jxq8aYUDRmykzVKrgoLhXSq67TEa5ruc4GJC2rWimw"
  , oauth_timestamp: "1272325550"
  , oauth_version: "1.0"
  , status: 'setting up my twitter 私のさえずりを設定する'
  }, "MCD8BKwGdgPHvAuvgvz4EQpqDAtx89grbuNMRd7Eh98", "J6zix3FfA9LofH0awS24M3HcBYXO5nI1iYe8EfBA")

console.log(upsign)
console.log('yOahq5m0YjDDjfjxHaXEsW9D+X0=')
assert.equal(upsign, 'yOahq5m0YjDDjfjxHaXEsW9D+X0=')


