function success(position) {
  var message = document.getElementById("status");
  message.innerHTML ="<img src='http://maps.google.com/maps/api/staticmap?center=" + position.coords.latitude + "," + position.coords.longitude + "&size=300x200&maptype=roadmap&zoom=12&&markers=size:mid|color:red|" + position.coords.latitude + "," + position.coords.longitude + "&sensor=false' />";
  message.innerHTML += "<p>Longitude: " + position.coords.longitude + "</p>";
  message.innerHTML += "<p>Latitude: " + position.coords.latitude + "</p>";
  message.innerHTML += "<p>Altitude: " + position.coords.altitude + "</p>";
}

function error(msg) {
  var message = document.getElementById("status");
  message.innerHTML = "Failed to get geolocation.";
}

if (navigator.geolocation) {
  navigator.geolocation.getCurrentPosition(success, error);
} else {
  error('Geolocation is not supported.');
}