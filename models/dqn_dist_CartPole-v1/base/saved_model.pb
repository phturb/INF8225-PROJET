??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
,
Exp
x"T
y"T"
Ttype:

2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

: *
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
: *
dtype0
?
categorical_dense_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3*+
shared_namecategorical_dense_0/kernel
?
.categorical_dense_0/kernel/Read/ReadVariableOpReadVariableOpcategorical_dense_0/kernel*
_output_shapes

: 3*
dtype0
?
categorical_dense_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:3*)
shared_namecategorical_dense_0/bias
?
,categorical_dense_0/bias/Read/ReadVariableOpReadVariableOpcategorical_dense_0/bias*
_output_shapes
:3*
dtype0
?
categorical_dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3*+
shared_namecategorical_dense_1/kernel
?
.categorical_dense_1/kernel/Read/ReadVariableOpReadVariableOpcategorical_dense_1/kernel*
_output_shapes

: 3*
dtype0
?
categorical_dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:3*)
shared_namecategorical_dense_1/bias
?
,categorical_dense_1/bias/Read/ReadVariableOpReadVariableOpcategorical_dense_1/bias*
_output_shapes
:3*
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
: *
dtype0
?
!Adam/categorical_dense_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3*2
shared_name#!Adam/categorical_dense_0/kernel/m
?
5Adam/categorical_dense_0/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/categorical_dense_0/kernel/m*
_output_shapes

: 3*
dtype0
?
Adam/categorical_dense_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:3*0
shared_name!Adam/categorical_dense_0/bias/m
?
3Adam/categorical_dense_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/categorical_dense_0/bias/m*
_output_shapes
:3*
dtype0
?
!Adam/categorical_dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3*2
shared_name#!Adam/categorical_dense_1/kernel/m
?
5Adam/categorical_dense_1/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/categorical_dense_1/kernel/m*
_output_shapes

: 3*
dtype0
?
Adam/categorical_dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:3*0
shared_name!Adam/categorical_dense_1/bias/m
?
3Adam/categorical_dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/categorical_dense_1/bias/m*
_output_shapes
:3*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
: *
dtype0
?
!Adam/categorical_dense_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3*2
shared_name#!Adam/categorical_dense_0/kernel/v
?
5Adam/categorical_dense_0/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/categorical_dense_0/kernel/v*
_output_shapes

: 3*
dtype0
?
Adam/categorical_dense_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:3*0
shared_name!Adam/categorical_dense_0/bias/v
?
3Adam/categorical_dense_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/categorical_dense_0/bias/v*
_output_shapes
:3*
dtype0
?
!Adam/categorical_dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3*2
shared_name#!Adam/categorical_dense_1/kernel/v
?
5Adam/categorical_dense_1/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/categorical_dense_1/kernel/v*
_output_shapes

: 3*
dtype0
?
Adam/categorical_dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:3*0
shared_name!Adam/categorical_dense_1/bias/v
?
3Adam/categorical_dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/categorical_dense_1/bias/v*
_output_shapes
:3*
dtype0

NoOpNoOp
?3
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?3
value?3B?3 B?3
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer-7
		optimizer

trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
h

!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api
R
'trainable_variables
(	variables
)regularization_losses
*	keras_api
R
+trainable_variables
,	variables
-regularization_losses
.	keras_api
R
/trainable_variables
0	variables
1regularization_losses
2	keras_api
?

3beta_1

4beta_2
	5decay
6learning_rate
7itermemfmgmhmimj!mk"mlvmvnvovpvqvr!vs"vt
8
0
1
2
3
4
5
!6
"7
8
0
1
2
3
4
5
!6
"7
 
?

trainable_variables
8metrics
9non_trainable_variables
	variables
:layer_metrics
;layer_regularization_losses

<layers
regularization_losses
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
=metrics
>non_trainable_variables
?layer_metrics
	variables
@layer_regularization_losses

Alayers
regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
Bmetrics
Cnon_trainable_variables
Dlayer_metrics
	variables
Elayer_regularization_losses

Flayers
regularization_losses
fd
VARIABLE_VALUEcategorical_dense_0/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEcategorical_dense_0/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
Gmetrics
Hnon_trainable_variables
Ilayer_metrics
	variables
Jlayer_regularization_losses

Klayers
regularization_losses
fd
VARIABLE_VALUEcategorical_dense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEcategorical_dense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
?
#trainable_variables
Lmetrics
Mnon_trainable_variables
Nlayer_metrics
$	variables
Olayer_regularization_losses

Players
%regularization_losses
 
 
 
?
'trainable_variables
Qmetrics
Rnon_trainable_variables
Slayer_metrics
(	variables
Tlayer_regularization_losses

Ulayers
)regularization_losses
 
 
 
?
+trainable_variables
Vmetrics
Wnon_trainable_variables
Xlayer_metrics
,	variables
Ylayer_regularization_losses

Zlayers
-regularization_losses
 
 
 
?
/trainable_variables
[metrics
\non_trainable_variables
]layer_metrics
0	variables
^layer_regularization_losses

_layers
1regularization_losses
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE

`0
 
 
 
8
0
1
2
3
4
5
6
7
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	atotal
	bcount
c	variables
d	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

a0
b1

c	variables
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/categorical_dense_0/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/categorical_dense_0/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/categorical_dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/categorical_dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/categorical_dense_0/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/categorical_dense_0/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/categorical_dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/categorical_dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasdense_1/kerneldense_1/biascategorical_dense_0/kernelcategorical_dense_0/biascategorical_dense_1/kernelcategorical_dense_1/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????3**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_2017406
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp.categorical_dense_0/kernel/Read/ReadVariableOp,categorical_dense_0/bias/Read/ReadVariableOp.categorical_dense_1/kernel/Read/ReadVariableOp,categorical_dense_1/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp5Adam/categorical_dense_0/kernel/m/Read/ReadVariableOp3Adam/categorical_dense_0/bias/m/Read/ReadVariableOp5Adam/categorical_dense_1/kernel/m/Read/ReadVariableOp3Adam/categorical_dense_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp5Adam/categorical_dense_0/kernel/v/Read/ReadVariableOp3Adam/categorical_dense_0/bias/v/Read/ReadVariableOp5Adam/categorical_dense_1/kernel/v/Read/ReadVariableOp3Adam/categorical_dense_1/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_2017785
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biascategorical_dense_0/kernelcategorical_dense_0/biascategorical_dense_1/kernelcategorical_dense_1/biasbeta_1beta_2decaylearning_rate	Adam/itertotalcountAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/m!Adam/categorical_dense_0/kernel/mAdam/categorical_dense_0/bias/m!Adam/categorical_dense_1/kernel/mAdam/categorical_dense_1/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v!Adam/categorical_dense_0/kernel/vAdam/categorical_dense_0/bias/v!Adam/categorical_dense_1/kernel/vAdam/categorical_dense_1/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_2017888??
?F
?
 __inference__traced_save_2017785
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop9
5savev2_categorical_dense_0_kernel_read_readvariableop7
3savev2_categorical_dense_0_bias_read_readvariableop9
5savev2_categorical_dense_1_kernel_read_readvariableop7
3savev2_categorical_dense_1_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop@
<savev2_adam_categorical_dense_0_kernel_m_read_readvariableop>
:savev2_adam_categorical_dense_0_bias_m_read_readvariableop@
<savev2_adam_categorical_dense_1_kernel_m_read_readvariableop>
:savev2_adam_categorical_dense_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop@
<savev2_adam_categorical_dense_0_kernel_v_read_readvariableop>
:savev2_adam_categorical_dense_0_bias_v_read_readvariableop@
<savev2_adam_categorical_dense_1_kernel_v_read_readvariableop>
:savev2_adam_categorical_dense_1_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop5savev2_categorical_dense_0_kernel_read_readvariableop3savev2_categorical_dense_0_bias_read_readvariableop5savev2_categorical_dense_1_kernel_read_readvariableop3savev2_categorical_dense_1_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop<savev2_adam_categorical_dense_0_kernel_m_read_readvariableop:savev2_adam_categorical_dense_0_bias_m_read_readvariableop<savev2_adam_categorical_dense_1_kernel_m_read_readvariableop:savev2_adam_categorical_dense_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop<savev2_adam_categorical_dense_0_kernel_v_read_readvariableop:savev2_adam_categorical_dense_0_bias_v_read_readvariableop<savev2_adam_categorical_dense_1_kernel_v_read_readvariableop:savev2_adam_categorical_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::: : : 3:3: 3:3: : : : : : : ::: : : 3:3: 3:3::: : : 3:3: 3:3: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: 3: 

_output_shapes
:3:$ 

_output_shapes

: 3: 

_output_shapes
:3:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: 3: 

_output_shapes
:3:$ 

_output_shapes

: 3: 

_output_shapes
:3:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: 3: 

_output_shapes
:3:$ 

_output_shapes

: 3: 

_output_shapes
:3: 

_output_shapes
: 
?	
?
B__inference_dense_layer_call_and_return_conditional_losses_2017099

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
1__inference_DQN_Categorical_layer_call_fn_2017327
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????3**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_DQN_Categorical_layer_call_and_return_conditional_losses_20173082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????32

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
P__inference_categorical_dense_0_layer_call_and_return_conditional_losses_2017594

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 3*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????32
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:3*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????32	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????32

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?>
?
L__inference_DQN_Categorical_layer_call_and_return_conditional_losses_2017454

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource6
2categorical_dense_0_matmul_readvariableop_resource7
3categorical_dense_0_biasadd_readvariableop_resource6
2categorical_dense_1_matmul_readvariableop_resource7
3categorical_dense_1_biasadd_readvariableop_resource
identity??*categorical_dense_0/BiasAdd/ReadVariableOp?)categorical_dense_0/MatMul/ReadVariableOp?*categorical_dense_1/BiasAdd/ReadVariableOp?)categorical_dense_1/MatMul/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_1/Relu?
)categorical_dense_0/MatMul/ReadVariableOpReadVariableOp2categorical_dense_0_matmul_readvariableop_resource*
_output_shapes

: 3*
dtype02+
)categorical_dense_0/MatMul/ReadVariableOp?
categorical_dense_0/MatMulMatMuldense_1/Relu:activations:01categorical_dense_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????32
categorical_dense_0/MatMul?
*categorical_dense_0/BiasAdd/ReadVariableOpReadVariableOp3categorical_dense_0_biasadd_readvariableop_resource*
_output_shapes
:3*
dtype02,
*categorical_dense_0/BiasAdd/ReadVariableOp?
categorical_dense_0/BiasAddBiasAdd$categorical_dense_0/MatMul:product:02categorical_dense_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????32
categorical_dense_0/BiasAdd?
)categorical_dense_1/MatMul/ReadVariableOpReadVariableOp2categorical_dense_1_matmul_readvariableop_resource*
_output_shapes

: 3*
dtype02+
)categorical_dense_1/MatMul/ReadVariableOp?
categorical_dense_1/MatMulMatMuldense_1/Relu:activations:01categorical_dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????32
categorical_dense_1/MatMul?
*categorical_dense_1/BiasAdd/ReadVariableOpReadVariableOp3categorical_dense_1_biasadd_readvariableop_resource*
_output_shapes
:3*
dtype02,
*categorical_dense_1/BiasAdd/ReadVariableOp?
categorical_dense_1/BiasAddBiasAdd$categorical_dense_1/MatMul:product:02categorical_dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????32
categorical_dense_1/BiasAddt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2$categorical_dense_0/BiasAdd:output:0$categorical_dense_1/BiasAdd:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????f2
concatenate/concati
reshape/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :32
reshape/Reshape/shape/2?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapeconcatenate/concat:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????32
reshape/Reshape?
 activation/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 activation/Max/reduction_indices?
activation/MaxMaxreshape/Reshape:output:0)activation/Max/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2
activation/Max?
activation/subSubreshape/Reshape:output:0activation/Max:output:0*
T0*+
_output_shapes
:?????????32
activation/subq
activation/ExpExpactivation/sub:z:0*
T0*+
_output_shapes
:?????????32
activation/Exp?
 activation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 activation/Sum/reduction_indices?
activation/SumSumactivation/Exp:y:0)activation/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2
activation/Sum?
activation/truedivRealDivactivation/Exp:y:0activation/Sum:output:0*
T0*+
_output_shapes
:?????????32
activation/truediv?
IdentityIdentityactivation/truediv:z:0+^categorical_dense_0/BiasAdd/ReadVariableOp*^categorical_dense_0/MatMul/ReadVariableOp+^categorical_dense_1/BiasAdd/ReadVariableOp*^categorical_dense_1/MatMul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????32

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2X
*categorical_dense_0/BiasAdd/ReadVariableOp*categorical_dense_0/BiasAdd/ReadVariableOp2V
)categorical_dense_0/MatMul/ReadVariableOp)categorical_dense_0/MatMul/ReadVariableOp2X
*categorical_dense_1/BiasAdd/ReadVariableOp*categorical_dense_1/BiasAdd/ReadVariableOp2V
)categorical_dense_1/MatMul/ReadVariableOp)categorical_dense_1/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_1_layer_call_and_return_conditional_losses_2017575

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_1_layer_call_and_return_conditional_losses_2017126

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
r
H__inference_concatenate_layer_call_and_return_conditional_losses_2017201

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????f2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????f2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????3:?????????3:O K
'
_output_shapes
:?????????3
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????3
 
_user_specified_nameinputs
?
Y
-__inference_concatenate_layer_call_fn_2017635
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????f* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_20172012
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????f2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????3:?????????3:Q M
'
_output_shapes
:?????????3
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????3
"
_user_specified_name
inputs/1
?
~
)__inference_dense_1_layer_call_fn_2017584

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_20171262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?>
?
L__inference_DQN_Categorical_layer_call_and_return_conditional_losses_2017502

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource6
2categorical_dense_0_matmul_readvariableop_resource7
3categorical_dense_0_biasadd_readvariableop_resource6
2categorical_dense_1_matmul_readvariableop_resource7
3categorical_dense_1_biasadd_readvariableop_resource
identity??*categorical_dense_0/BiasAdd/ReadVariableOp?)categorical_dense_0/MatMul/ReadVariableOp?*categorical_dense_1/BiasAdd/ReadVariableOp?)categorical_dense_1/MatMul/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_1/Relu?
)categorical_dense_0/MatMul/ReadVariableOpReadVariableOp2categorical_dense_0_matmul_readvariableop_resource*
_output_shapes

: 3*
dtype02+
)categorical_dense_0/MatMul/ReadVariableOp?
categorical_dense_0/MatMulMatMuldense_1/Relu:activations:01categorical_dense_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????32
categorical_dense_0/MatMul?
*categorical_dense_0/BiasAdd/ReadVariableOpReadVariableOp3categorical_dense_0_biasadd_readvariableop_resource*
_output_shapes
:3*
dtype02,
*categorical_dense_0/BiasAdd/ReadVariableOp?
categorical_dense_0/BiasAddBiasAdd$categorical_dense_0/MatMul:product:02categorical_dense_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????32
categorical_dense_0/BiasAdd?
)categorical_dense_1/MatMul/ReadVariableOpReadVariableOp2categorical_dense_1_matmul_readvariableop_resource*
_output_shapes

: 3*
dtype02+
)categorical_dense_1/MatMul/ReadVariableOp?
categorical_dense_1/MatMulMatMuldense_1/Relu:activations:01categorical_dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????32
categorical_dense_1/MatMul?
*categorical_dense_1/BiasAdd/ReadVariableOpReadVariableOp3categorical_dense_1_biasadd_readvariableop_resource*
_output_shapes
:3*
dtype02,
*categorical_dense_1/BiasAdd/ReadVariableOp?
categorical_dense_1/BiasAddBiasAdd$categorical_dense_1/MatMul:product:02categorical_dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????32
categorical_dense_1/BiasAddt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2$categorical_dense_0/BiasAdd:output:0$categorical_dense_1/BiasAdd:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????f2
concatenate/concati
reshape/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :32
reshape/Reshape/shape/2?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapeconcatenate/concat:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????32
reshape/Reshape?
 activation/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 activation/Max/reduction_indices?
activation/MaxMaxreshape/Reshape:output:0)activation/Max/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2
activation/Max?
activation/subSubreshape/Reshape:output:0activation/Max:output:0*
T0*+
_output_shapes
:?????????32
activation/subq
activation/ExpExpactivation/sub:z:0*
T0*+
_output_shapes
:?????????32
activation/Exp?
 activation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 activation/Sum/reduction_indices?
activation/SumSumactivation/Exp:y:0)activation/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2
activation/Sum?
activation/truedivRealDivactivation/Exp:y:0activation/Sum:output:0*
T0*+
_output_shapes
:?????????32
activation/truediv?
IdentityIdentityactivation/truediv:z:0+^categorical_dense_0/BiasAdd/ReadVariableOp*^categorical_dense_0/MatMul/ReadVariableOp+^categorical_dense_1/BiasAdd/ReadVariableOp*^categorical_dense_1/MatMul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????32

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2X
*categorical_dense_0/BiasAdd/ReadVariableOp*categorical_dense_0/BiasAdd/ReadVariableOp2V
)categorical_dense_0/MatMul/ReadVariableOp)categorical_dense_0/MatMul/ReadVariableOp2X
*categorical_dense_1/BiasAdd/ReadVariableOp*categorical_dense_1/BiasAdd/ReadVariableOp2V
)categorical_dense_1/MatMul/ReadVariableOp)categorical_dense_1/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_reshape_layer_call_and_return_conditional_losses_2017223

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :32
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????32	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????32

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????f:O K
'
_output_shapes
:?????????f
 
_user_specified_nameinputs
?Q
?
"__inference__wrapped_model_2017084
input_18
4dqn_categorical_dense_matmul_readvariableop_resource9
5dqn_categorical_dense_biasadd_readvariableop_resource:
6dqn_categorical_dense_1_matmul_readvariableop_resource;
7dqn_categorical_dense_1_biasadd_readvariableop_resourceF
Bdqn_categorical_categorical_dense_0_matmul_readvariableop_resourceG
Cdqn_categorical_categorical_dense_0_biasadd_readvariableop_resourceF
Bdqn_categorical_categorical_dense_1_matmul_readvariableop_resourceG
Cdqn_categorical_categorical_dense_1_biasadd_readvariableop_resource
identity??:DQN_Categorical/categorical_dense_0/BiasAdd/ReadVariableOp?9DQN_Categorical/categorical_dense_0/MatMul/ReadVariableOp?:DQN_Categorical/categorical_dense_1/BiasAdd/ReadVariableOp?9DQN_Categorical/categorical_dense_1/MatMul/ReadVariableOp?,DQN_Categorical/dense/BiasAdd/ReadVariableOp?+DQN_Categorical/dense/MatMul/ReadVariableOp?.DQN_Categorical/dense_1/BiasAdd/ReadVariableOp?-DQN_Categorical/dense_1/MatMul/ReadVariableOp?
+DQN_Categorical/dense/MatMul/ReadVariableOpReadVariableOp4dqn_categorical_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+DQN_Categorical/dense/MatMul/ReadVariableOp?
DQN_Categorical/dense/MatMulMatMulinput_13DQN_Categorical/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
DQN_Categorical/dense/MatMul?
,DQN_Categorical/dense/BiasAdd/ReadVariableOpReadVariableOp5dqn_categorical_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,DQN_Categorical/dense/BiasAdd/ReadVariableOp?
DQN_Categorical/dense/BiasAddBiasAdd&DQN_Categorical/dense/MatMul:product:04DQN_Categorical/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
DQN_Categorical/dense/BiasAdd?
DQN_Categorical/dense/ReluRelu&DQN_Categorical/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
DQN_Categorical/dense/Relu?
-DQN_Categorical/dense_1/MatMul/ReadVariableOpReadVariableOp6dqn_categorical_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02/
-DQN_Categorical/dense_1/MatMul/ReadVariableOp?
DQN_Categorical/dense_1/MatMulMatMul(DQN_Categorical/dense/Relu:activations:05DQN_Categorical/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2 
DQN_Categorical/dense_1/MatMul?
.DQN_Categorical/dense_1/BiasAdd/ReadVariableOpReadVariableOp7dqn_categorical_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.DQN_Categorical/dense_1/BiasAdd/ReadVariableOp?
DQN_Categorical/dense_1/BiasAddBiasAdd(DQN_Categorical/dense_1/MatMul:product:06DQN_Categorical/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2!
DQN_Categorical/dense_1/BiasAdd?
DQN_Categorical/dense_1/ReluRelu(DQN_Categorical/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
DQN_Categorical/dense_1/Relu?
9DQN_Categorical/categorical_dense_0/MatMul/ReadVariableOpReadVariableOpBdqn_categorical_categorical_dense_0_matmul_readvariableop_resource*
_output_shapes

: 3*
dtype02;
9DQN_Categorical/categorical_dense_0/MatMul/ReadVariableOp?
*DQN_Categorical/categorical_dense_0/MatMulMatMul*DQN_Categorical/dense_1/Relu:activations:0ADQN_Categorical/categorical_dense_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????32,
*DQN_Categorical/categorical_dense_0/MatMul?
:DQN_Categorical/categorical_dense_0/BiasAdd/ReadVariableOpReadVariableOpCdqn_categorical_categorical_dense_0_biasadd_readvariableop_resource*
_output_shapes
:3*
dtype02<
:DQN_Categorical/categorical_dense_0/BiasAdd/ReadVariableOp?
+DQN_Categorical/categorical_dense_0/BiasAddBiasAdd4DQN_Categorical/categorical_dense_0/MatMul:product:0BDQN_Categorical/categorical_dense_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????32-
+DQN_Categorical/categorical_dense_0/BiasAdd?
9DQN_Categorical/categorical_dense_1/MatMul/ReadVariableOpReadVariableOpBdqn_categorical_categorical_dense_1_matmul_readvariableop_resource*
_output_shapes

: 3*
dtype02;
9DQN_Categorical/categorical_dense_1/MatMul/ReadVariableOp?
*DQN_Categorical/categorical_dense_1/MatMulMatMul*DQN_Categorical/dense_1/Relu:activations:0ADQN_Categorical/categorical_dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????32,
*DQN_Categorical/categorical_dense_1/MatMul?
:DQN_Categorical/categorical_dense_1/BiasAdd/ReadVariableOpReadVariableOpCdqn_categorical_categorical_dense_1_biasadd_readvariableop_resource*
_output_shapes
:3*
dtype02<
:DQN_Categorical/categorical_dense_1/BiasAdd/ReadVariableOp?
+DQN_Categorical/categorical_dense_1/BiasAddBiasAdd4DQN_Categorical/categorical_dense_1/MatMul:product:0BDQN_Categorical/categorical_dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????32-
+DQN_Categorical/categorical_dense_1/BiasAdd?
'DQN_Categorical/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2)
'DQN_Categorical/concatenate/concat/axis?
"DQN_Categorical/concatenate/concatConcatV24DQN_Categorical/categorical_dense_0/BiasAdd:output:04DQN_Categorical/categorical_dense_1/BiasAdd:output:00DQN_Categorical/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????f2$
"DQN_Categorical/concatenate/concat?
DQN_Categorical/reshape/ShapeShape+DQN_Categorical/concatenate/concat:output:0*
T0*
_output_shapes
:2
DQN_Categorical/reshape/Shape?
+DQN_Categorical/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+DQN_Categorical/reshape/strided_slice/stack?
-DQN_Categorical/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-DQN_Categorical/reshape/strided_slice/stack_1?
-DQN_Categorical/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-DQN_Categorical/reshape/strided_slice/stack_2?
%DQN_Categorical/reshape/strided_sliceStridedSlice&DQN_Categorical/reshape/Shape:output:04DQN_Categorical/reshape/strided_slice/stack:output:06DQN_Categorical/reshape/strided_slice/stack_1:output:06DQN_Categorical/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%DQN_Categorical/reshape/strided_slice?
'DQN_Categorical/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'DQN_Categorical/reshape/Reshape/shape/1?
'DQN_Categorical/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :32)
'DQN_Categorical/reshape/Reshape/shape/2?
%DQN_Categorical/reshape/Reshape/shapePack.DQN_Categorical/reshape/strided_slice:output:00DQN_Categorical/reshape/Reshape/shape/1:output:00DQN_Categorical/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2'
%DQN_Categorical/reshape/Reshape/shape?
DQN_Categorical/reshape/ReshapeReshape+DQN_Categorical/concatenate/concat:output:0.DQN_Categorical/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????32!
DQN_Categorical/reshape/Reshape?
0DQN_Categorical/activation/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0DQN_Categorical/activation/Max/reduction_indices?
DQN_Categorical/activation/MaxMax(DQN_Categorical/reshape/Reshape:output:09DQN_Categorical/activation/Max/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2 
DQN_Categorical/activation/Max?
DQN_Categorical/activation/subSub(DQN_Categorical/reshape/Reshape:output:0'DQN_Categorical/activation/Max:output:0*
T0*+
_output_shapes
:?????????32 
DQN_Categorical/activation/sub?
DQN_Categorical/activation/ExpExp"DQN_Categorical/activation/sub:z:0*
T0*+
_output_shapes
:?????????32 
DQN_Categorical/activation/Exp?
0DQN_Categorical/activation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0DQN_Categorical/activation/Sum/reduction_indices?
DQN_Categorical/activation/SumSum"DQN_Categorical/activation/Exp:y:09DQN_Categorical/activation/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2 
DQN_Categorical/activation/Sum?
"DQN_Categorical/activation/truedivRealDiv"DQN_Categorical/activation/Exp:y:0'DQN_Categorical/activation/Sum:output:0*
T0*+
_output_shapes
:?????????32$
"DQN_Categorical/activation/truediv?
IdentityIdentity&DQN_Categorical/activation/truediv:z:0;^DQN_Categorical/categorical_dense_0/BiasAdd/ReadVariableOp:^DQN_Categorical/categorical_dense_0/MatMul/ReadVariableOp;^DQN_Categorical/categorical_dense_1/BiasAdd/ReadVariableOp:^DQN_Categorical/categorical_dense_1/MatMul/ReadVariableOp-^DQN_Categorical/dense/BiasAdd/ReadVariableOp,^DQN_Categorical/dense/MatMul/ReadVariableOp/^DQN_Categorical/dense_1/BiasAdd/ReadVariableOp.^DQN_Categorical/dense_1/MatMul/ReadVariableOp*
T0*+
_output_shapes
:?????????32

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2x
:DQN_Categorical/categorical_dense_0/BiasAdd/ReadVariableOp:DQN_Categorical/categorical_dense_0/BiasAdd/ReadVariableOp2v
9DQN_Categorical/categorical_dense_0/MatMul/ReadVariableOp9DQN_Categorical/categorical_dense_0/MatMul/ReadVariableOp2x
:DQN_Categorical/categorical_dense_1/BiasAdd/ReadVariableOp:DQN_Categorical/categorical_dense_1/BiasAdd/ReadVariableOp2v
9DQN_Categorical/categorical_dense_1/MatMul/ReadVariableOp9DQN_Categorical/categorical_dense_1/MatMul/ReadVariableOp2\
,DQN_Categorical/dense/BiasAdd/ReadVariableOp,DQN_Categorical/dense/BiasAdd/ReadVariableOp2Z
+DQN_Categorical/dense/MatMul/ReadVariableOp+DQN_Categorical/dense/MatMul/ReadVariableOp2`
.DQN_Categorical/dense_1/BiasAdd/ReadVariableOp.DQN_Categorical/dense_1/BiasAdd/ReadVariableOp2^
-DQN_Categorical/dense_1/MatMul/ReadVariableOp-DQN_Categorical/dense_1/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
1__inference_DQN_Categorical_layer_call_fn_2017375
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????3**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_DQN_Categorical_layer_call_and_return_conditional_losses_20173562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????32

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
5__inference_categorical_dense_0_layer_call_fn_2017603

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????3*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_categorical_dense_0_layer_call_and_return_conditional_losses_20171522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????32

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?!
?
L__inference_DQN_Categorical_layer_call_and_return_conditional_losses_2017251
input_1
dense_2017110
dense_2017112
dense_1_2017137
dense_1_2017139
categorical_dense_0_2017163
categorical_dense_0_2017165
categorical_dense_1_2017189
categorical_dense_1_2017191
identity??+categorical_dense_0/StatefulPartitionedCall?+categorical_dense_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_2017110dense_2017112*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_20170992
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2017137dense_1_2017139*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_20171262!
dense_1/StatefulPartitionedCall?
+categorical_dense_0/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0categorical_dense_0_2017163categorical_dense_0_2017165*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????3*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_categorical_dense_0_layer_call_and_return_conditional_losses_20171522-
+categorical_dense_0/StatefulPartitionedCall?
+categorical_dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0categorical_dense_1_2017189categorical_dense_1_2017191*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????3*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_categorical_dense_1_layer_call_and_return_conditional_losses_20171782-
+categorical_dense_1/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall4categorical_dense_0/StatefulPartitionedCall:output:04categorical_dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????f* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_20172012
concatenate/PartitionedCall?
reshape/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_20172232
reshape/PartitionedCall?
activation/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_20172422
activation/PartitionedCall?
IdentityIdentity#activation/PartitionedCall:output:0,^categorical_dense_0/StatefulPartitionedCall,^categorical_dense_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????32

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2Z
+categorical_dense_0/StatefulPartitionedCall+categorical_dense_0/StatefulPartitionedCall2Z
+categorical_dense_1/StatefulPartitionedCall+categorical_dense_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
c
G__inference_activation_layer_call_and_return_conditional_losses_2017242

inputs
identityy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Max/reduction_indices?
MaxMaxinputsMax/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2
Max]
subSubinputsMax:output:0*
T0*+
_output_shapes
:?????????32
subP
ExpExpsub:z:0*
T0*+
_output_shapes
:?????????32
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indices?
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2
Sumj
truedivRealDivExp:y:0Sum:output:0*
T0*+
_output_shapes
:?????????32	
truedivc
IdentityIdentitytruediv:z:0*
T0*+
_output_shapes
:?????????32

Identity"
identityIdentity:output:0**
_input_shapes
:?????????3:S O
+
_output_shapes
:?????????3
 
_user_specified_nameinputs
?
?
5__inference_categorical_dense_1_layer_call_fn_2017622

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????3*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_categorical_dense_1_layer_call_and_return_conditional_losses_20171782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????32

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
E
)__inference_reshape_layer_call_fn_2017653

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_20172232
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????32

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????f:O K
'
_output_shapes
:?????????f
 
_user_specified_nameinputs
?	
?
P__inference_categorical_dense_0_layer_call_and_return_conditional_losses_2017152

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 3*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????32
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:3*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????32	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????32

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?!
?
L__inference_DQN_Categorical_layer_call_and_return_conditional_losses_2017308

inputs
dense_2017284
dense_2017286
dense_1_2017289
dense_1_2017291
categorical_dense_0_2017294
categorical_dense_0_2017296
categorical_dense_1_2017299
categorical_dense_1_2017301
identity??+categorical_dense_0/StatefulPartitionedCall?+categorical_dense_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2017284dense_2017286*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_20170992
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2017289dense_1_2017291*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_20171262!
dense_1/StatefulPartitionedCall?
+categorical_dense_0/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0categorical_dense_0_2017294categorical_dense_0_2017296*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????3*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_categorical_dense_0_layer_call_and_return_conditional_losses_20171522-
+categorical_dense_0/StatefulPartitionedCall?
+categorical_dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0categorical_dense_1_2017299categorical_dense_1_2017301*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????3*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_categorical_dense_1_layer_call_and_return_conditional_losses_20171782-
+categorical_dense_1/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall4categorical_dense_0/StatefulPartitionedCall:output:04categorical_dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????f* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_20172012
concatenate/PartitionedCall?
reshape/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_20172232
reshape/PartitionedCall?
activation/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_20172422
activation/PartitionedCall?
IdentityIdentity#activation/PartitionedCall:output:0,^categorical_dense_0/StatefulPartitionedCall,^categorical_dense_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????32

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2Z
+categorical_dense_0/StatefulPartitionedCall+categorical_dense_0/StatefulPartitionedCall2Z
+categorical_dense_1/StatefulPartitionedCall+categorical_dense_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
1__inference_DQN_Categorical_layer_call_fn_2017523

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????3**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_DQN_Categorical_layer_call_and_return_conditional_losses_20173082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????32

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
L__inference_DQN_Categorical_layer_call_and_return_conditional_losses_2017278
input_1
dense_2017254
dense_2017256
dense_1_2017259
dense_1_2017261
categorical_dense_0_2017264
categorical_dense_0_2017266
categorical_dense_1_2017269
categorical_dense_1_2017271
identity??+categorical_dense_0/StatefulPartitionedCall?+categorical_dense_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_2017254dense_2017256*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_20170992
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2017259dense_1_2017261*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_20171262!
dense_1/StatefulPartitionedCall?
+categorical_dense_0/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0categorical_dense_0_2017264categorical_dense_0_2017266*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????3*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_categorical_dense_0_layer_call_and_return_conditional_losses_20171522-
+categorical_dense_0/StatefulPartitionedCall?
+categorical_dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0categorical_dense_1_2017269categorical_dense_1_2017271*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????3*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_categorical_dense_1_layer_call_and_return_conditional_losses_20171782-
+categorical_dense_1/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall4categorical_dense_0/StatefulPartitionedCall:output:04categorical_dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????f* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_20172012
concatenate/PartitionedCall?
reshape/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_20172232
reshape/PartitionedCall?
activation/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_20172422
activation/PartitionedCall?
IdentityIdentity#activation/PartitionedCall:output:0,^categorical_dense_0/StatefulPartitionedCall,^categorical_dense_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????32

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2Z
+categorical_dense_0/StatefulPartitionedCall+categorical_dense_0/StatefulPartitionedCall2Z
+categorical_dense_1/StatefulPartitionedCall+categorical_dense_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
1__inference_DQN_Categorical_layer_call_fn_2017544

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????3**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_DQN_Categorical_layer_call_and_return_conditional_losses_20173562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????32

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
L__inference_DQN_Categorical_layer_call_and_return_conditional_losses_2017356

inputs
dense_2017332
dense_2017334
dense_1_2017337
dense_1_2017339
categorical_dense_0_2017342
categorical_dense_0_2017344
categorical_dense_1_2017347
categorical_dense_1_2017349
identity??+categorical_dense_0/StatefulPartitionedCall?+categorical_dense_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2017332dense_2017334*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_20170992
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2017337dense_1_2017339*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_20171262!
dense_1/StatefulPartitionedCall?
+categorical_dense_0/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0categorical_dense_0_2017342categorical_dense_0_2017344*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????3*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_categorical_dense_0_layer_call_and_return_conditional_losses_20171522-
+categorical_dense_0/StatefulPartitionedCall?
+categorical_dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0categorical_dense_1_2017347categorical_dense_1_2017349*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????3*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_categorical_dense_1_layer_call_and_return_conditional_losses_20171782-
+categorical_dense_1/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall4categorical_dense_0/StatefulPartitionedCall:output:04categorical_dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????f* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_20172012
concatenate/PartitionedCall?
reshape/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_20172232
reshape/PartitionedCall?
activation/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_20172422
activation/PartitionedCall?
IdentityIdentity#activation/PartitionedCall:output:0,^categorical_dense_0/StatefulPartitionedCall,^categorical_dense_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????32

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2Z
+categorical_dense_0/StatefulPartitionedCall+categorical_dense_0/StatefulPartitionedCall2Z
+categorical_dense_1/StatefulPartitionedCall+categorical_dense_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_layer_call_and_return_conditional_losses_2017555

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
c
G__inference_activation_layer_call_and_return_conditional_losses_2017664

inputs
identityy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Max/reduction_indices?
MaxMaxinputsMax/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2
Max]
subSubinputsMax:output:0*
T0*+
_output_shapes
:?????????32
subP
ExpExpsub:z:0*
T0*+
_output_shapes
:?????????32
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indices?
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2
Sumj
truedivRealDivExp:y:0Sum:output:0*
T0*+
_output_shapes
:?????????32	
truedivc
IdentityIdentitytruediv:z:0*
T0*+
_output_shapes
:?????????32

Identity"
identityIdentity:output:0**
_input_shapes
:?????????3:S O
+
_output_shapes
:?????????3
 
_user_specified_nameinputs
?
|
'__inference_dense_layer_call_fn_2017564

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_20170992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_reshape_layer_call_and_return_conditional_losses_2017648

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :32
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????32	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????32

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????f:O K
'
_output_shapes
:?????????f
 
_user_specified_nameinputs
?
H
,__inference_activation_layer_call_fn_2017669

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_20172422
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????32

Identity"
identityIdentity:output:0**
_input_shapes
:?????????3:S O
+
_output_shapes
:?????????3
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_2017406
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????3**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_20170842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????32

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1

?
#__inference__traced_restore_2017888
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias1
-assignvariableop_4_categorical_dense_0_kernel/
+assignvariableop_5_categorical_dense_0_bias1
-assignvariableop_6_categorical_dense_1_kernel/
+assignvariableop_7_categorical_dense_1_bias
assignvariableop_8_beta_1
assignvariableop_9_beta_2
assignvariableop_10_decay%
!assignvariableop_11_learning_rate!
assignvariableop_12_adam_iter
assignvariableop_13_total
assignvariableop_14_count+
'assignvariableop_15_adam_dense_kernel_m)
%assignvariableop_16_adam_dense_bias_m-
)assignvariableop_17_adam_dense_1_kernel_m+
'assignvariableop_18_adam_dense_1_bias_m9
5assignvariableop_19_adam_categorical_dense_0_kernel_m7
3assignvariableop_20_adam_categorical_dense_0_bias_m9
5assignvariableop_21_adam_categorical_dense_1_kernel_m7
3assignvariableop_22_adam_categorical_dense_1_bias_m+
'assignvariableop_23_adam_dense_kernel_v)
%assignvariableop_24_adam_dense_bias_v-
)assignvariableop_25_adam_dense_1_kernel_v+
'assignvariableop_26_adam_dense_1_bias_v9
5assignvariableop_27_adam_categorical_dense_0_kernel_v7
3assignvariableop_28_adam_categorical_dense_0_bias_v9
5assignvariableop_29_adam_categorical_dense_1_kernel_v7
3assignvariableop_30_adam_categorical_dense_1_bias_v
identity_32??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::*.
dtypes$
"2 	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp-assignvariableop_4_categorical_dense_0_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp+assignvariableop_5_categorical_dense_0_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp-assignvariableop_6_categorical_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp+assignvariableop_7_categorical_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_beta_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_beta_2Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_decayIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_dense_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp%assignvariableop_16_adam_dense_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp5assignvariableop_19_adam_categorical_dense_0_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp3assignvariableop_20_adam_categorical_dense_0_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp5assignvariableop_21_adam_categorical_dense_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp3assignvariableop_22_adam_categorical_dense_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_dense_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_1_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp5assignvariableop_27_adam_categorical_dense_0_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp3assignvariableop_28_adam_categorical_dense_0_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp5assignvariableop_29_adam_categorical_dense_1_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp3assignvariableop_30_adam_categorical_dense_1_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_309
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_31?
Identity_32IdentityIdentity_31:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_32"#
identity_32Identity_32:output:0*?
_input_shapes?
~: :::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
t
H__inference_concatenate_layer_call_and_return_conditional_losses_2017629
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????f2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????f2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????3:?????????3:Q M
'
_output_shapes
:?????????3
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????3
"
_user_specified_name
inputs/1
?	
?
P__inference_categorical_dense_1_layer_call_and_return_conditional_losses_2017178

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 3*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????32
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:3*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????32	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????32

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
P__inference_categorical_dense_1_layer_call_and_return_conditional_losses_2017613

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 3*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????32
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:3*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????32	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????32

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????B

activation4
StatefulPartitionedCall:0?????????3tensorflow/serving/predict:??
?9
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer-7
		optimizer

trainable_variables
	variables
regularization_losses
	keras_api

signatures
u__call__
v_default_save_signature
*w&call_and_return_all_conditional_losses"?6
_tf_keras_network?6{"class_name": "Functional", "name": "DQN_Categorical", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "DQN_Categorical", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "categorical_dense_0", "trainable": true, "dtype": "float32", "units": 51, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "categorical_dense_0", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "categorical_dense_1", "trainable": true, "dtype": "float32", "units": 51, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "categorical_dense_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["categorical_dense_0", 0, 0, {}], ["categorical_dense_1", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 51]}}, "name": "reshape", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "softmax"}, "name": "activation", "inbound_nodes": [[["reshape", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["activation", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 4]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "DQN_Categorical", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "categorical_dense_0", "trainable": true, "dtype": "float32", "units": 51, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "categorical_dense_0", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "categorical_dense_1", "trainable": true, "dtype": "float32", "units": 51, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "categorical_dense_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["categorical_dense_0", 0, 0, {}], ["categorical_dense_1", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 51]}}, "name": "reshape", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "softmax"}, "name": "activation", "inbound_nodes": [[["reshape", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["activation", 0, 0]]}}, "training_config": {"loss": "modified_KL_Divergence", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
x__call__
*y&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
z__call__
*{&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
|__call__
*}&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "categorical_dense_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "categorical_dense_0", "trainable": true, "dtype": "float32", "units": 51, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?

!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api
~__call__
*&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "categorical_dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "categorical_dense_1", "trainable": true, "dtype": "float32", "units": 51, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
'trainable_variables
(	variables
)regularization_losses
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 51]}, {"class_name": "TensorShape", "items": [null, 51]}]}
?
+trainable_variables
,	variables
-regularization_losses
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 51]}}}
?
/trainable_variables
0	variables
1regularization_losses
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "softmax"}}
?

3beta_1

4beta_2
	5decay
6learning_rate
7itermemfmgmhmimj!mk"mlvmvnvovpvqvr!vs"vt"
	optimizer
X
0
1
2
3
4
5
!6
"7"
trackable_list_wrapper
X
0
1
2
3
4
5
!6
"7"
trackable_list_wrapper
 "
trackable_list_wrapper
?

trainable_variables
8metrics
9non_trainable_variables
	variables
:layer_metrics
;layer_regularization_losses

<layers
regularization_losses
u__call__
v_default_save_signature
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
:2dense/kernel
:2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
=metrics
>non_trainable_variables
?layer_metrics
	variables
@layer_regularization_losses

Alayers
regularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 : 2dense_1/kernel
: 2dense_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
Bmetrics
Cnon_trainable_variables
Dlayer_metrics
	variables
Elayer_regularization_losses

Flayers
regularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
,:* 32categorical_dense_0/kernel
&:$32categorical_dense_0/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
Gmetrics
Hnon_trainable_variables
Ilayer_metrics
	variables
Jlayer_regularization_losses

Klayers
regularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
,:* 32categorical_dense_1/kernel
&:$32categorical_dense_1/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
#trainable_variables
Lmetrics
Mnon_trainable_variables
Nlayer_metrics
$	variables
Olayer_regularization_losses

Players
%regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
'trainable_variables
Qmetrics
Rnon_trainable_variables
Slayer_metrics
(	variables
Tlayer_regularization_losses

Ulayers
)regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
+trainable_variables
Vmetrics
Wnon_trainable_variables
Xlayer_metrics
,	variables
Ylayer_regularization_losses

Zlayers
-regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
/trainable_variables
[metrics
\non_trainable_variables
]layer_metrics
0	variables
^layer_regularization_losses

_layers
1regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
'
`0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	atotal
	bcount
c	variables
d	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
a0
b1"
trackable_list_wrapper
-
c	variables"
_generic_user_object
#:!2Adam/dense/kernel/m
:2Adam/dense/bias/m
%:# 2Adam/dense_1/kernel/m
: 2Adam/dense_1/bias/m
1:/ 32!Adam/categorical_dense_0/kernel/m
+:)32Adam/categorical_dense_0/bias/m
1:/ 32!Adam/categorical_dense_1/kernel/m
+:)32Adam/categorical_dense_1/bias/m
#:!2Adam/dense/kernel/v
:2Adam/dense/bias/v
%:# 2Adam/dense_1/kernel/v
: 2Adam/dense_1/bias/v
1:/ 32!Adam/categorical_dense_0/kernel/v
+:)32Adam/categorical_dense_0/bias/v
1:/ 32!Adam/categorical_dense_1/kernel/v
+:)32Adam/categorical_dense_1/bias/v
?2?
1__inference_DQN_Categorical_layer_call_fn_2017523
1__inference_DQN_Categorical_layer_call_fn_2017327
1__inference_DQN_Categorical_layer_call_fn_2017375
1__inference_DQN_Categorical_layer_call_fn_2017544?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
"__inference__wrapped_model_2017084?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
L__inference_DQN_Categorical_layer_call_and_return_conditional_losses_2017502
L__inference_DQN_Categorical_layer_call_and_return_conditional_losses_2017251
L__inference_DQN_Categorical_layer_call_and_return_conditional_losses_2017278
L__inference_DQN_Categorical_layer_call_and_return_conditional_losses_2017454?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dense_layer_call_fn_2017564?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_layer_call_and_return_conditional_losses_2017555?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_1_layer_call_fn_2017584?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_1_layer_call_and_return_conditional_losses_2017575?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_categorical_dense_0_layer_call_fn_2017603?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_categorical_dense_0_layer_call_and_return_conditional_losses_2017594?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_categorical_dense_1_layer_call_fn_2017622?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_categorical_dense_1_layer_call_and_return_conditional_losses_2017613?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_concatenate_layer_call_fn_2017635?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_concatenate_layer_call_and_return_conditional_losses_2017629?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_reshape_layer_call_fn_2017653?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_reshape_layer_call_and_return_conditional_losses_2017648?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_activation_layer_call_fn_2017669?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_activation_layer_call_and_return_conditional_losses_2017664?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_2017406input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
L__inference_DQN_Categorical_layer_call_and_return_conditional_losses_2017251o!"8?5
.?+
!?
input_1?????????
p

 
? ")?&
?
0?????????3
? ?
L__inference_DQN_Categorical_layer_call_and_return_conditional_losses_2017278o!"8?5
.?+
!?
input_1?????????
p 

 
? ")?&
?
0?????????3
? ?
L__inference_DQN_Categorical_layer_call_and_return_conditional_losses_2017454n!"7?4
-?*
 ?
inputs?????????
p

 
? ")?&
?
0?????????3
? ?
L__inference_DQN_Categorical_layer_call_and_return_conditional_losses_2017502n!"7?4
-?*
 ?
inputs?????????
p 

 
? ")?&
?
0?????????3
? ?
1__inference_DQN_Categorical_layer_call_fn_2017327b!"8?5
.?+
!?
input_1?????????
p

 
? "??????????3?
1__inference_DQN_Categorical_layer_call_fn_2017375b!"8?5
.?+
!?
input_1?????????
p 

 
? "??????????3?
1__inference_DQN_Categorical_layer_call_fn_2017523a!"7?4
-?*
 ?
inputs?????????
p

 
? "??????????3?
1__inference_DQN_Categorical_layer_call_fn_2017544a!"7?4
-?*
 ?
inputs?????????
p 

 
? "??????????3?
"__inference__wrapped_model_2017084y!"0?-
&?#
!?
input_1?????????
? ";?8
6

activation(?%

activation?????????3?
G__inference_activation_layer_call_and_return_conditional_losses_2017664`3?0
)?&
$?!
inputs?????????3
? ")?&
?
0?????????3
? ?
,__inference_activation_layer_call_fn_2017669S3?0
)?&
$?!
inputs?????????3
? "??????????3?
P__inference_categorical_dense_0_layer_call_and_return_conditional_losses_2017594\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????3
? ?
5__inference_categorical_dense_0_layer_call_fn_2017603O/?,
%?"
 ?
inputs????????? 
? "??????????3?
P__inference_categorical_dense_1_layer_call_and_return_conditional_losses_2017613\!"/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????3
? ?
5__inference_categorical_dense_1_layer_call_fn_2017622O!"/?,
%?"
 ?
inputs????????? 
? "??????????3?
H__inference_concatenate_layer_call_and_return_conditional_losses_2017629?Z?W
P?M
K?H
"?
inputs/0?????????3
"?
inputs/1?????????3
? "%?"
?
0?????????f
? ?
-__inference_concatenate_layer_call_fn_2017635vZ?W
P?M
K?H
"?
inputs/0?????????3
"?
inputs/1?????????3
? "??????????f?
D__inference_dense_1_layer_call_and_return_conditional_losses_2017575\/?,
%?"
 ?
inputs?????????
? "%?"
?
0????????? 
? |
)__inference_dense_1_layer_call_fn_2017584O/?,
%?"
 ?
inputs?????????
? "?????????? ?
B__inference_dense_layer_call_and_return_conditional_losses_2017555\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_layer_call_fn_2017564O/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_reshape_layer_call_and_return_conditional_losses_2017648\/?,
%?"
 ?
inputs?????????f
? ")?&
?
0?????????3
? |
)__inference_reshape_layer_call_fn_2017653O/?,
%?"
 ?
inputs?????????f
? "??????????3?
%__inference_signature_wrapper_2017406?!";?8
? 
1?.
,
input_1!?
input_1?????????";?8
6

activation(?%

activation?????????3