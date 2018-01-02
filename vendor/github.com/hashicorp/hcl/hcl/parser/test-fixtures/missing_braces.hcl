# should error, but not crash
resource "template_file" "cloud_config" {
  template = "$file("${path.module}/some/path")"
}
