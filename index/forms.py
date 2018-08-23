from django import forms

from index.models import *


class UploadFileForm(forms.ModelForm):
    class Meta:
        model = UploadFileModel
        fields = ('name', 'file')
