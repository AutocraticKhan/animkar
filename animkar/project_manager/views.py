from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import ListView, CreateView, UpdateView, DeleteView, DetailView
from django.urls import reverse_lazy
from .models import Project
from .forms import ProjectForm

class ProjectListView(ListView):
    model = Project
    template_name = 'project_manager/project_list.html'
    context_object_name = 'projects'
    ordering = ['-created_at']

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = ProjectForm()
        return context

class ProjectDetailView(DetailView):
    model = Project
    template_name = 'project_manager/project_detail.html'
    context_object_name = 'project'

class ProjectCreateView(CreateView):
    model = Project
    form_class = ProjectForm
    template_name = 'project_manager/project_form.html'
    success_url = reverse_lazy('project_list')

class ProjectUpdateView(UpdateView):
    model = Project
    form_class = ProjectForm
    template_name = 'project_manager/project_form.html'
    success_url = reverse_lazy('project_list')

class ProjectDeleteView(DeleteView):
    model = Project
    template_name = 'project_manager/project_confirm_delete.html'
    success_url = reverse_lazy('project_list')
