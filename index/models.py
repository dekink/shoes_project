from django.db import models

# Create your models here.
class UploadFileModel(models.Model):
    name = models.CharField(default='', null=True, max_length=10)
    file = models.FileField(null=True)
