provisioner "remote-exec" {
  scripts = [
    "${path.module}/scripts/install-consul.sh" // missing comma
    "${path.module}/scripts/install-haproxy.sh"
  ] 
}
