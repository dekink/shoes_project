from django.shortcuts import render
from index.forms import UploadFileForm
from django.http import HttpResponseRedirect
from index.models import UploadFileModel

def index(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            #return HttpResponseRedirect('/upload')
            return upload(request, form)
    else:
        form = UploadFileForm()
    return render(request, 'index/index.html', {'form': form})

def upload(request, file):
    length = len(UploadFileModel.objects.all())
    img_info = UploadFileModel.objects.all()[length - 1]
    return render(request, 'index/upload.html', {'info': img_info})
