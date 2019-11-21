function varargout = nldr_interactive(varargin)
% NLDR_INTERACTIVE MATLAB code for nldr_interactive.fig
%      NLDR_INTERACTIVE, by itself, creates a new NLDR_INTERACTIVE or raises the existing
%      singleton*.
%
%      H = NLDR_INTERACTIVE returns the handle to a new NLDR_INTERACTIVE or the handle to
%      the existing singleton*.
%
%      NLDR_INTERACTIVE('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in NLDR_INTERACTIVE.M with the given input arguments.
%auto
%      NLDR_INTERACTIVE('Property','Value',...) creates a new NLDR_INTERACTIVE or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before nldr_interactive_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to nldr_interactive_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help nldr_interactive

% Last Modified by GUIDE v2.5 18-Jul-2019 16:20:11

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @nldr_interactive_OpeningFcn, ...
    'gui_OutputFcn',  @nldr_interactive_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

function plot_original(handles)
    data=handles.data.data;
    axes(handles.axes_figure);
    scatter3(data(:,1),data(:,2),data(:,3),10,handles.cmap);
    if ~isfield(handles.data,'NN')
        handles.data.NN=6;
    end

% --- Executes just before nldr_interactive is made visible.
function nldr_interactive_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to nldr_interactive (see VARARGIN)

% Add current dir and sub dir to PATH
% addpath(genpath(pwd));rmpath(genpath([pwd '\.git']));

% Choose default command line output for nldr_interactive
handles.output = hObject;
handles.fname=handles.listbox_FILE.String{handles.listbox_FILE.Value};
handles.data=load(handles.fname);
N=size(handles.data.data,1);
handles.cmap=jet(N);
plot_original(handles);
handles.method=handles.listbox_METHOD.String{handles.listbox_METHOD.Value};
handles.nn=handles.data.NN;
set(handles.edit_KNN,'String',num2str(handles.nn));
% Update handles structure
guidata(hObject, handles);

% UIWAIT makes nldr_interactive wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = nldr_interactive_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


function listbox_FILE_Callback(hObject, eventdata, handles)
handles.fname=hObject.String{hObject.Value};
handles.data=load(handles.fname);
N=size(handles.data.data,1);
handles.cmap=jet(N);
plot_original(handles);
handles.edit_KNN.String=handles.data.NN;
handles.nn=handles.data.NN;
guidata(hObject,handles);

function listbox_FILE_CreateFcn(hObject, eventdata, handles)
% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
a=dir('data/NLDR*.mat');
hObject.String={a.name};

function listbox_METHOD_Callback(hObject, eventdata, handles)
% set(handles.edit_KNN,'String',num2str(6));
% handles.nn=6;
handles.method=hObject.String{hObject.Value};
guidata(hObject,handles);

function listbox_METHOD_CreateFcn(hObject, eventdata, handles)

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
% hObject.String={'PCA','MDS','LLE','ISOMAP','LTSA','LE','HESSIAN'};
% hObject.String={'PCA','LLE','LTSA','LE','HSIC-LE','HSICT-LE','HSIC-LTSA','HESSIAN','HLE','ISOMAP'};
hObject.String={'PCA','LLE','LTSA','LE','HSIC-LE','HSICT-LE','HESSIAN','ISOMAP'};


function edit_KNN_Callback(hObject, eventdata, handles)
handles.nn=str2double(hObject.String);
guidata(hObject,handles);

function edit_KNN_CreateFcn(hObject, eventdata, handles)

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function pushbutton_RUN_Callback(hObject, eventdata, handles)
mapX=experimentnldr(handles.fname,handles.method,2,handles.nn);
handles.mapX=mapX;
axes(handles.axes_figure);
scatter(mapX(:,1),mapX(:,2),10,handles.cmap);
axis square;
guidata(hObject, handles);

function edit_T_Callback(hObject, eventdata, handles)
handles.t=str2double(hObject.String);
guidata(hObject,handles);

function edit_T_CreateFcn(hObject, eventdata, handles)

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function pushbutton_fig_Callback(hObject, eventdata, handles)
fig=figure;
if isfield(handles,'mapX')
    mapX=handles.mapX;
    scatter(mapX(:,1),mapX(:,2),10,handles.cmap);
    xlabel('\phi_{1}'); ylabel('\phi_{2}');
    %title([handles.method ' NN=' num2str(handles.nn) ' t=' num2str(handles.t)]);
    axis square;
    fname=['0' num2str(handles.listbox_FILE.Value) '_' num2str(handles.listbox_METHOD.Value+1) '_'...
        char(regexprep(handles.listbox_FILE.String(handles.listbox_FILE.Value),'.mat','','ignorecase'))...
        '_' char(handles.listbox_METHOD.String(handles.listbox_METHOD.Value))...
        '_NN' num2str(handles.nn)];
else
    mapX=handles.data.data;
    scatter3(mapX(:,1),mapX(:,2),mapX(:,3),10,handles.cmap);
    xlabel('x'); ylabel('y');zlabel('z');%title(inputdlg('Title'));
    axis square;
    fname=['0' num2str(handles.listbox_FILE.Value) '_' num2str(1) '_'...
        char(regexprep(handles.listbox_FILE.String(handles.listbox_FILE.Value),'.mat','','ignorecase'))];
end
ChangeFigProperties(fig,fname);


function pushbutton_compare_Callback(hObject, eventdata, handles)
    tm = 1e-10;
    methods=get(handles.listbox_METHOD,'String');
    fig=figure;
    mapX=handles.data.data;
    scatter3(mapX(:,1),mapX(:,2),mapX(:,3),10,handles.cmap);
    xlabel('x'); ylabel('y');zlabel('z');%title(handles.fname);
    axis square;
    fname=['0' num2str(handles.listbox_FILE.Value) '_'...
        char(regexprep(handles.fname,'.mat','','ignorecase'))];
    ChangeFigProperties(fig,fname);
    for i=1:length(methods)
        pause(tm);
        mymsgbox = msgbox(['Computing ' methods{i}], 'Processing');
        tic;
        mapX=experimentnldr(handles.fname,methods{i},2,handles.nn);
        fig=figure;
        scatter(mapX(:,1),mapX(:,2),10,handles.cmap);
        xlabel('\phi_{1}'); ylabel('\phi_{2}');
        %title([methods{i} ' NN=' num2str(handles.nn)]);
        axis square;
        fname=['0' num2str(handles.listbox_FILE.Value) '_'...
            char(regexprep(handles.fname,'.mat','','ignorecase'))...
            '_' methods{i} '_NN' num2str(handles.nn)];
        ChangeFigProperties(fig,fname);
        close(mymsgbox);
        mymsgbox = msgbox(['Completed ' methods{i} 'in ' num2str(toc) ' seconds'], 'Success');
        pause(tm)
        close(mymsgbox);
    end


function pushbutton_reset_Callback(hObject, eventdata, handles)
    close all;clear;clc;
    nldr_interactive;