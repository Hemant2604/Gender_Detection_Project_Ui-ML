from django.db import models
from django.core.files.storage import FileSystemStorage

# Create your models here.

class ImageUpload(models.Model):
    image = models.ImageField(upload_to='uploads/')
    result_image = models.ImageField(upload_to='results/', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Image uploaded at {self.created_at}"

