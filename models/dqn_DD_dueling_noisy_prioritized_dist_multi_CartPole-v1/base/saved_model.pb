å
¶
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
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

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
¾
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
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

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8§
|
noisy_dense/w_muVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namenoisy_dense/w_mu
u
$noisy_dense/w_mu/Read/ReadVariableOpReadVariableOpnoisy_dense/w_mu*
_output_shapes

:*
dtype0

noisy_dense/w_sigmaVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_namenoisy_dense/w_sigma
{
'noisy_dense/w_sigma/Read/ReadVariableOpReadVariableOpnoisy_dense/w_sigma*
_output_shapes

:*
dtype0
x
noisy_dense/b_muVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namenoisy_dense/b_mu
q
$noisy_dense/b_mu/Read/ReadVariableOpReadVariableOpnoisy_dense/b_mu*
_output_shapes
:*
dtype0
~
noisy_dense/b_sigmaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namenoisy_dense/b_sigma
w
'noisy_dense/b_sigma/Read/ReadVariableOpReadVariableOpnoisy_dense/b_sigma*
_output_shapes
:*
dtype0

noisy_dense_1/w_muVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *#
shared_namenoisy_dense_1/w_mu
y
&noisy_dense_1/w_mu/Read/ReadVariableOpReadVariableOpnoisy_dense_1/w_mu*
_output_shapes

: *
dtype0

noisy_dense_1/w_sigmaVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_namenoisy_dense_1/w_sigma

)noisy_dense_1/w_sigma/Read/ReadVariableOpReadVariableOpnoisy_dense_1/w_sigma*
_output_shapes

: *
dtype0
|
noisy_dense_1/b_muVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namenoisy_dense_1/b_mu
u
&noisy_dense_1/b_mu/Read/ReadVariableOpReadVariableOpnoisy_dense_1/b_mu*
_output_shapes
: *
dtype0

noisy_dense_1/b_sigmaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namenoisy_dense_1/b_sigma
{
)noisy_dense_1/b_sigma/Read/ReadVariableOpReadVariableOpnoisy_dense_1/b_sigma*
_output_shapes
: *
dtype0

noisy_dense_2/w_muVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3*#
shared_namenoisy_dense_2/w_mu
y
&noisy_dense_2/w_mu/Read/ReadVariableOpReadVariableOpnoisy_dense_2/w_mu*
_output_shapes

: 3*
dtype0

noisy_dense_2/w_sigmaVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3*&
shared_namenoisy_dense_2/w_sigma

)noisy_dense_2/w_sigma/Read/ReadVariableOpReadVariableOpnoisy_dense_2/w_sigma*
_output_shapes

: 3*
dtype0
|
noisy_dense_2/b_muVarHandleOp*
_output_shapes
: *
dtype0*
shape:3*#
shared_namenoisy_dense_2/b_mu
u
&noisy_dense_2/b_mu/Read/ReadVariableOpReadVariableOpnoisy_dense_2/b_mu*
_output_shapes
:3*
dtype0

noisy_dense_2/b_sigmaVarHandleOp*
_output_shapes
: *
dtype0*
shape:3*&
shared_namenoisy_dense_2/b_sigma
{
)noisy_dense_2/b_sigma/Read/ReadVariableOpReadVariableOpnoisy_dense_2/b_sigma*
_output_shapes
:3*
dtype0

noisy_dense_3/w_muVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3*#
shared_namenoisy_dense_3/w_mu
y
&noisy_dense_3/w_mu/Read/ReadVariableOpReadVariableOpnoisy_dense_3/w_mu*
_output_shapes

: 3*
dtype0

noisy_dense_3/w_sigmaVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3*&
shared_namenoisy_dense_3/w_sigma

)noisy_dense_3/w_sigma/Read/ReadVariableOpReadVariableOpnoisy_dense_3/w_sigma*
_output_shapes

: 3*
dtype0
|
noisy_dense_3/b_muVarHandleOp*
_output_shapes
: *
dtype0*
shape:3*#
shared_namenoisy_dense_3/b_mu
u
&noisy_dense_3/b_mu/Read/ReadVariableOpReadVariableOpnoisy_dense_3/b_mu*
_output_shapes
:3*
dtype0

noisy_dense_3/b_sigmaVarHandleOp*
_output_shapes
: *
dtype0*
shape:3*&
shared_namenoisy_dense_3/b_sigma
{
)noisy_dense_3/b_sigma/Read/ReadVariableOpReadVariableOpnoisy_dense_3/b_sigma*
_output_shapes
:3*
dtype0

categorical_dense/w_muVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3*'
shared_namecategorical_dense/w_mu

*categorical_dense/w_mu/Read/ReadVariableOpReadVariableOpcategorical_dense/w_mu*
_output_shapes

: 3*
dtype0

categorical_dense/w_sigmaVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3**
shared_namecategorical_dense/w_sigma

-categorical_dense/w_sigma/Read/ReadVariableOpReadVariableOpcategorical_dense/w_sigma*
_output_shapes

: 3*
dtype0

categorical_dense/b_muVarHandleOp*
_output_shapes
: *
dtype0*
shape:3*'
shared_namecategorical_dense/b_mu
}
*categorical_dense/b_mu/Read/ReadVariableOpReadVariableOpcategorical_dense/b_mu*
_output_shapes
:3*
dtype0

categorical_dense/b_sigmaVarHandleOp*
_output_shapes
: *
dtype0*
shape:3**
shared_namecategorical_dense/b_sigma

-categorical_dense/b_sigma/Read/ReadVariableOpReadVariableOpcategorical_dense/b_sigma*
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

Adam/noisy_dense/w_mu/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/noisy_dense/w_mu/m

+Adam/noisy_dense/w_mu/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense/w_mu/m*
_output_shapes

:*
dtype0

Adam/noisy_dense/w_sigma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/noisy_dense/w_sigma/m

.Adam/noisy_dense/w_sigma/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense/w_sigma/m*
_output_shapes

:*
dtype0

Adam/noisy_dense/b_mu/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/noisy_dense/b_mu/m

+Adam/noisy_dense/b_mu/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense/b_mu/m*
_output_shapes
:*
dtype0

Adam/noisy_dense/b_sigma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/noisy_dense/b_sigma/m

.Adam/noisy_dense/b_sigma/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense/b_sigma/m*
_output_shapes
:*
dtype0

Adam/noisy_dense_1/w_mu/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: **
shared_nameAdam/noisy_dense_1/w_mu/m

-Adam/noisy_dense_1/w_mu/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_1/w_mu/m*
_output_shapes

: *
dtype0

Adam/noisy_dense_1/w_sigma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *-
shared_nameAdam/noisy_dense_1/w_sigma/m

0Adam/noisy_dense_1/w_sigma/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_1/w_sigma/m*
_output_shapes

: *
dtype0

Adam/noisy_dense_1/b_mu/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/noisy_dense_1/b_mu/m

-Adam/noisy_dense_1/b_mu/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_1/b_mu/m*
_output_shapes
: *
dtype0

Adam/noisy_dense_1/b_sigma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/noisy_dense_1/b_sigma/m

0Adam/noisy_dense_1/b_sigma/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_1/b_sigma/m*
_output_shapes
: *
dtype0

Adam/noisy_dense_2/w_mu/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3**
shared_nameAdam/noisy_dense_2/w_mu/m

-Adam/noisy_dense_2/w_mu/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_2/w_mu/m*
_output_shapes

: 3*
dtype0

Adam/noisy_dense_2/w_sigma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3*-
shared_nameAdam/noisy_dense_2/w_sigma/m

0Adam/noisy_dense_2/w_sigma/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_2/w_sigma/m*
_output_shapes

: 3*
dtype0

Adam/noisy_dense_2/b_mu/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:3**
shared_nameAdam/noisy_dense_2/b_mu/m

-Adam/noisy_dense_2/b_mu/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_2/b_mu/m*
_output_shapes
:3*
dtype0

Adam/noisy_dense_2/b_sigma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:3*-
shared_nameAdam/noisy_dense_2/b_sigma/m

0Adam/noisy_dense_2/b_sigma/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_2/b_sigma/m*
_output_shapes
:3*
dtype0

Adam/noisy_dense_3/w_mu/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3**
shared_nameAdam/noisy_dense_3/w_mu/m

-Adam/noisy_dense_3/w_mu/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_3/w_mu/m*
_output_shapes

: 3*
dtype0

Adam/noisy_dense_3/w_sigma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3*-
shared_nameAdam/noisy_dense_3/w_sigma/m

0Adam/noisy_dense_3/w_sigma/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_3/w_sigma/m*
_output_shapes

: 3*
dtype0

Adam/noisy_dense_3/b_mu/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:3**
shared_nameAdam/noisy_dense_3/b_mu/m

-Adam/noisy_dense_3/b_mu/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_3/b_mu/m*
_output_shapes
:3*
dtype0

Adam/noisy_dense_3/b_sigma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:3*-
shared_nameAdam/noisy_dense_3/b_sigma/m

0Adam/noisy_dense_3/b_sigma/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_3/b_sigma/m*
_output_shapes
:3*
dtype0

Adam/categorical_dense/w_mu/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3*.
shared_nameAdam/categorical_dense/w_mu/m

1Adam/categorical_dense/w_mu/m/Read/ReadVariableOpReadVariableOpAdam/categorical_dense/w_mu/m*
_output_shapes

: 3*
dtype0

 Adam/categorical_dense/w_sigma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3*1
shared_name" Adam/categorical_dense/w_sigma/m

4Adam/categorical_dense/w_sigma/m/Read/ReadVariableOpReadVariableOp Adam/categorical_dense/w_sigma/m*
_output_shapes

: 3*
dtype0

Adam/categorical_dense/b_mu/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:3*.
shared_nameAdam/categorical_dense/b_mu/m

1Adam/categorical_dense/b_mu/m/Read/ReadVariableOpReadVariableOpAdam/categorical_dense/b_mu/m*
_output_shapes
:3*
dtype0

 Adam/categorical_dense/b_sigma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:3*1
shared_name" Adam/categorical_dense/b_sigma/m

4Adam/categorical_dense/b_sigma/m/Read/ReadVariableOpReadVariableOp Adam/categorical_dense/b_sigma/m*
_output_shapes
:3*
dtype0

Adam/noisy_dense/w_mu/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/noisy_dense/w_mu/v

+Adam/noisy_dense/w_mu/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense/w_mu/v*
_output_shapes

:*
dtype0

Adam/noisy_dense/w_sigma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/noisy_dense/w_sigma/v

.Adam/noisy_dense/w_sigma/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense/w_sigma/v*
_output_shapes

:*
dtype0

Adam/noisy_dense/b_mu/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/noisy_dense/b_mu/v

+Adam/noisy_dense/b_mu/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense/b_mu/v*
_output_shapes
:*
dtype0

Adam/noisy_dense/b_sigma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/noisy_dense/b_sigma/v

.Adam/noisy_dense/b_sigma/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense/b_sigma/v*
_output_shapes
:*
dtype0

Adam/noisy_dense_1/w_mu/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: **
shared_nameAdam/noisy_dense_1/w_mu/v

-Adam/noisy_dense_1/w_mu/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_1/w_mu/v*
_output_shapes

: *
dtype0

Adam/noisy_dense_1/w_sigma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *-
shared_nameAdam/noisy_dense_1/w_sigma/v

0Adam/noisy_dense_1/w_sigma/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_1/w_sigma/v*
_output_shapes

: *
dtype0

Adam/noisy_dense_1/b_mu/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/noisy_dense_1/b_mu/v

-Adam/noisy_dense_1/b_mu/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_1/b_mu/v*
_output_shapes
: *
dtype0

Adam/noisy_dense_1/b_sigma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/noisy_dense_1/b_sigma/v

0Adam/noisy_dense_1/b_sigma/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_1/b_sigma/v*
_output_shapes
: *
dtype0

Adam/noisy_dense_2/w_mu/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3**
shared_nameAdam/noisy_dense_2/w_mu/v

-Adam/noisy_dense_2/w_mu/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_2/w_mu/v*
_output_shapes

: 3*
dtype0

Adam/noisy_dense_2/w_sigma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3*-
shared_nameAdam/noisy_dense_2/w_sigma/v

0Adam/noisy_dense_2/w_sigma/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_2/w_sigma/v*
_output_shapes

: 3*
dtype0

Adam/noisy_dense_2/b_mu/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:3**
shared_nameAdam/noisy_dense_2/b_mu/v

-Adam/noisy_dense_2/b_mu/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_2/b_mu/v*
_output_shapes
:3*
dtype0

Adam/noisy_dense_2/b_sigma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:3*-
shared_nameAdam/noisy_dense_2/b_sigma/v

0Adam/noisy_dense_2/b_sigma/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_2/b_sigma/v*
_output_shapes
:3*
dtype0

Adam/noisy_dense_3/w_mu/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3**
shared_nameAdam/noisy_dense_3/w_mu/v

-Adam/noisy_dense_3/w_mu/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_3/w_mu/v*
_output_shapes

: 3*
dtype0

Adam/noisy_dense_3/w_sigma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3*-
shared_nameAdam/noisy_dense_3/w_sigma/v

0Adam/noisy_dense_3/w_sigma/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_3/w_sigma/v*
_output_shapes

: 3*
dtype0

Adam/noisy_dense_3/b_mu/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:3**
shared_nameAdam/noisy_dense_3/b_mu/v

-Adam/noisy_dense_3/b_mu/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_3/b_mu/v*
_output_shapes
:3*
dtype0

Adam/noisy_dense_3/b_sigma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:3*-
shared_nameAdam/noisy_dense_3/b_sigma/v

0Adam/noisy_dense_3/b_sigma/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_3/b_sigma/v*
_output_shapes
:3*
dtype0

Adam/categorical_dense/w_mu/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3*.
shared_nameAdam/categorical_dense/w_mu/v

1Adam/categorical_dense/w_mu/v/Read/ReadVariableOpReadVariableOpAdam/categorical_dense/w_mu/v*
_output_shapes

: 3*
dtype0

 Adam/categorical_dense/w_sigma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 3*1
shared_name" Adam/categorical_dense/w_sigma/v

4Adam/categorical_dense/w_sigma/v/Read/ReadVariableOpReadVariableOp Adam/categorical_dense/w_sigma/v*
_output_shapes

: 3*
dtype0

Adam/categorical_dense/b_mu/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:3*.
shared_nameAdam/categorical_dense/b_mu/v

1Adam/categorical_dense/b_mu/v/Read/ReadVariableOpReadVariableOpAdam/categorical_dense/b_mu/v*
_output_shapes
:3*
dtype0

 Adam/categorical_dense/b_sigma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:3*1
shared_name" Adam/categorical_dense/b_sigma/v

4Adam/categorical_dense/b_sigma/v/Read/ReadVariableOpReadVariableOp Adam/categorical_dense/b_sigma/v*
_output_shapes
:3*
dtype0
Ú
ConstConst*
_output_shapes

:*
dtype0*
valueB"Æ.?ç¦A?2$@¿ð¿Íñ)?c{L¿±?8 ¿æMt?Õ;È>{?W)¿Í?5z?ì_G¿iÄt¿è"?ÂäT?¡;S¿jÃ­¿Ô:?ºÌ`¿24?ý0¿J?!Ü>§m)?*:¿(?,#?Y/[¿<¿ÀÉT¿f
¿¿ô?(øâ?©	t¿Ñ?Å¿äëe?h¯¿*Ä¿N]¿|+s?r;¿WjÅ¿F&?±½¯?Ê/¿ [e¿ÿc?3»?ñFI¿¦.r?Jå¢¿b¢=?\¬¿&í¾6¿³H?×¿òÒ¢¿!"l?ò?

Const_1Const*
_output_shapes
:*
dtype0*U
valueLBJ"@æÊ5?+m?À¸k¿}èÁ¿}P?~Üz¿¼»¨?#nD¿¡Û?D¦õ>=?³¿O¿0$ ?¼¨¨?zt¿S$¿
Ü
Const_2Const*
_output_shapes

: *
dtype0*
valueB "Ä¿¿Ý¾.¿Ç~?|ü³>ÂßY¿^x¿Jb^?1@¿¤:m¿2?íÂ|¿ÇÎ?=N?¾£¡?@Ô¿ã-¿?gf¿Á`Ñ?ë\¾¿{.?8ÏÓ?=Ho¿wi??º?ºØL?D«?K
?±;¿Ê^¿üµ¿6ó|?xÈ>ÑÊi?f¿fÞ¢¾9'E?¿`?ô;I¿6ê-?àªV?Ò!¿ ¹d?R´¿Fé:¿Útj>^)¿!?/hm?ª}P?w½¿1B¬?ã¿Wª¿¿|X?Ú¯x¿J'¿_]9¿2¿ÿdt¿b×)?9Íö>B¨~?u¥¯¿ï
¿ãW¢¿R ?t0â>Èæ¿2¿;¼?¬q¿0¿wW`?²Ò¿ûÇ?Ê?õÍ¢¾ý±?Ö6¿tÚ¤¿6Æ¿@Î:ï¿7E[?>@sZ¿ ¯¬?¹]³?6·?.	²?´©?ißk¿~`+¿ðÔ°¿=éP?é>¥>´A?ñh>¿X¾Ô"?9?ô2&¿Ô¢?'K1?j¿×æ<?Ø^m¿¦^¿#£A>l²S¿zØ>÷D?<1,?z¿¢D? f¿ÒK¿öÓ2?âcM¿ÙUU¿¬¿ÐÀS¿WØI¿E?HÕË>1RR?.ïâ¾z3¾2¿Ñ¾FÖÎ>(>Jà°¾¢É¾´>!¾ÍÀ¾ì>Ü2Í¾ªì ?°§>¾WÒ½äõå>
'k¾eýÔ¾8»¾ú)?¹¿ð¥>æó+?AÂ¾ß>½ç>ÓL¦>æ>9BÛ>_¾k]¾Fwä¾¡:î>o<>§/Ü>¨!Ù¾åc¾î­¹>%«Ó>Ü½¾"Ë£>¥,Ê>"#¾^i×>VW¿°¾ËÏÜ=ægñ¾3Ûv>,ß>u[Ä>Rp2¿Ë;"?º²¾Ï4¿ìË>È6ê¾0Fó¾¶®¾Oxñ¾à+æ¾ÿô>>ph>=Öï>S~j¿&{¹¾Ç»X¿ºU?&ü>Ä6¿vYP¿:?¡9!¿G¿lÀ?¨T¿8?ïE-?hYY¾×m?Cüò¾¡\¿<GA¿¤¯?°¿Ñ]?2®±?û¹H¿f?¡uo?ëÖ+?ÿ®m?êb?är¿.Ëä¾{l¿-^¾óø½ s¾on>±¦Ê=YOu¾Ò¾Dcz>:eX¾¾ìþH>K¾AÎ²>öh>ÉÜ½fw>#¾´²¾_µ¾u¾ë>£UÖ¾ÆsD>O{î>û´¾5·>X³ >[¤f>>>>­SS¾ó¾n¾<Kl¿¸çº¾ÇeZ¿"^W?ë$>Ð+8¿ûòQ¿Àû;?v"¿9H¿Äæ?l©U¿å=?.?[¾åqo?ÝÙô¾7Æ]¿"ÃB¿Cý°?nê ¿}?p³?DJ¿¼Oh?MLq?®(-?-o?<Md?^¨¿ãæ¾ãm¿y!?r>L>?B,¿úïO¾µû>Wx?«u ¿L
Þ>ï	?'=Î¾Ô?Mx7¿¢î¾Øª>M #¿R§>??¼äq¿òì[?ªÉ¾Þ³t¿8
?bÀ¿}ä$¿¨ì¾m«#¿ë¿æÖØ>K>"?ëõ°?/ù?Ý£?ûI¡¿¼áã¾í?%;?çÇ¿VVs?º-?4b¿î ?GÉ¿½Â¿Ñ¤>R³¿{^7?>¦?Û?!Àñ?=é\¿/Àuz?jú­¿Oµ´¿Å­¿8^³¿«ùª¿=£m?Æ¨,?«'²?õ]:?éi>@,?ÅÜ)¿³þo¾ãA?¶%?C¿Ï" ?S)?×î¾T(?\ÁS¿¶	¿Î½,>.Ú<¿ºÁ>Qê.?U?Î¿ÎÔ}?Y§è¾Î6¿¾?ß97¿YP>¿i¿ç<¿C4¿õDú>IÖµ>ö;?cñÖ¾P*¾ç©Æ¾XèÃ>e
>¦§¾ú¾¾gÿª>È¾i¶¾.D>[Â¾Z9ô>©Ó>d:Ç½)ÏÙ>º^¾;¼É¾û)±¾9ÿ ?3`¿Ô)>ÛÝ"?Áý·¾RÓ>³~Û>>>ùÝÙ>C¬Ï>^R¾æ·Q¾ÄdØ¾=!¿WVê¾.é¿ð?<Á¾>Ðèf¿Y¿k°k?8±K¿%l{¿W2=?*ñ¿&O¨?ðéZ?ÁL¾í?~¿¿40t¿­çÝ?ÀÉ¿yë8?c{à?0}¿ ¢?SD?@Y?"%? ?×ëF¿¿-!¿¦¿R½¿Öï¿ Í?zÖ>þÏ¿eû¿á?ée¿.X¿¹T?O¿e=½?8#v?Ë_¾¢Å¨?,¿"Q¿G¿0ù?×â¿êO?ðeü?W¿x¾£?ª?Þt?Ñ¨?òê ?¥¨_¿±"¿Ó¬§¿Zp®¿ú	¿1:¡¿ý?h¢à>Üõ¿ý¿SÆ? Þo¿Ý	¿«Ì^?2»¿Å3Æ?·å?t¯¡¾ÛÃ°?Á4¿X¸£¿oÇ¿¨@Îí¿WÃY?-@ÛQ¿»«?"²?d©?ßÏ°?ñ¨?R@j¿ç2*¿¿¯¿
Ô
Const_3Const*
_output_shapes
: *
dtype0*
valueB "¨z?¾Ô>x?«#u¿-­¾Y¡Q?Ãøn?2øU¿Øë8?À@d?ðÂ+¿2s?Ì¿w½F¿UKy>¾E¿bY?xn|?y¯]?¤tÉ¿þ(·?÷à'¿ËË¿v:f?q6¿¼S¿E¿O¿>î¿4?Û5?c?
Ü3
Const_4Const*
_output_shapes

: 3*
dtype0*3
value3B3 3"3Wð\¿ëM¿¶£Ð?±X©¾üê?ÆS?8Þ¿r¿pÃ¿)Ý¿èn?&î?øw ¿eë1?JÌ§¿¥Ñ>q>xE¿HÍU?4\?4x?ý3Z¿,.É¿A©¬?+C?c¿h_¿Om|¿ù%?^¿m@5¾Ætý>Î42?á$ ¾èlf?jª¿U½N¿Ð<£?h»{¿¹d¿&·?Ö¿¡æ[?>z¿ÔX?|n?l?¨§P¿g¸Ð>³¿ð! ?¡?äåÂ?]Àê÷>ajR¿N½¿n[?î°?lÜ?è¬!@úb®¿K!Þ¿ê?X¿ZTõ?ãA¿0×¾ìZ?xK¿4À¿zqµ¿?q@pü¿k¿Ñc¦?âNÊ?à¸?ñ©r¿þ®¢?î>zH9¿F¿vZ;>qr¨¿gã?Ï!?r©î¿Ó¸?;º¦?¦½¿>@ÛÀ ¿ùÙ?³½¿¯V®¿<Ä¿@?~¿öE?Lê¿²óf¿X¿^Ú?w±¾¼p?=0?©¿¥ÿ|¿¿{/ç¿\y?ÜÐ?¾½§¿¢û9?g¯¿ñ%Û>¤>+kN¿Õ}_?BP?ã¹?d¿ILÒ¿||´?­ØK?Iím¿Ò¤¿ÿî¿2-? h¿Sw=¾³x?_H:?ó¾SÞp?w¸¢¿òX¿¶¢ª?¿Úhn¿?Íß¿ñÝe?å4¿5?Ky?¢?~Z¿ÿ-Ú>U=¿Ðc§?ýw?îV?¼é¿·½>Ô:!¿Oâ¿Kå'?[?æ¨?Ä÷?r¿º4ª¿{Å³?KRG¿û»?Ýê¾\-¥¾9]?&o¿W)¿ª¿dst?_aá?:nÁ¿OwZ¿¤ý~?}?Ne?ð9¿Oy?ïK>âø¿¤G¿û>X¿d®?¹g?ß¶¿§??¤Q¿ Úï?PZv¿V¦?ùç¿¿|¿Ái?ÊÓé¾^?e³¿òï±>ê·Ö>"(¿éb>dÐg¾òOÐ¾íeq>UìÂ>#×ò>2?À¾H¸ô¾<?uJ¾©#?Ô×(¾R}í½	>|0¬¾9Ó¾7åÇ¾ç»¯>@"?R¿Ä¾¨O·>ÓáÞ>ÿKË>Â«¾V:³>wù=/ L¾¾hN=Ö¹¾¼ú>l¦>jw¿¹¼Ê>Û®·>ðÐ¾ïm,?ò±¾(ï>XÐ¾vÀ¾ÎØ¾P¨>Í(¾ò¢Y>?÷ ¿à>T?-?!nÈ¿ç®¢>UA
¿-zx¿¢ø?®h?ÕÔ?ÑuÔ?*e¿Êó¿'?@ë*¿2¡?ßeÉ¾Ý£¾³=?´cM¿b|¿þon¿Q?ºCÁ?Þ¥¿åU;¿×§Z?í?«~r?·q¿ùÈU?¸.>Ä{ó¾Æ1+¿;4ö=â[]¿n?äF?Ð¿ÄÓq?e[?9y¿ÿ¬Í??S¿©¢?åx¿äe¿"¿ìqH?È¾µÌ?öÔ¿q;q¿h¿Íã?©æ¸¾#?¾4?.¢#¿O!¿s¤¿ãyñ¿;?â¥?05¯¿óBB?Ù5·¿.çä>éû >ZW¿pi?ùl?$?Ã>n¿­¨Û¿*¼?ëT?x¿¿hÎ¿;85?\ûr¿UæE¾=^
?B?ê¾{?³ö©¿ºa¿$;²?Im¿y¿?¡?ÞÃé¿Rp?¢¿C:?á1?ÒÄ?êÑc¿3äã>É¿A×®?{%¿ÿÄ²¿9æ@¬ã¾À A?o­?ûH¿ÙI¢¿æ.Ê¿ÕKÀô?~¿Ë?Ð2×¿én?á¿%?lºÅ>Æh¿a\?)°?m¦?ßO¿ÒåÀ§ç?aÂ?Þ¿¹¿B©¿#^?8¿ás¾Fó)?\ün?gÙ+¾é?èÁÐ¿ ¿kéÚ?SË¨¿!î¿Þô­?ÀPs?Ç¿av­?Wé?¹D´?ßè¿ô?õ25¿q¿Ö?+÷p?7c?"ã¿U²¸>ö¿Ç¿Þs#?êû?Ým¤?5ñ?4¿§³¥¿¯?øB¿ÿ·?f¦ä¾YÎ ¾U^W?.i¿bD¿ËY¿Vûm?jÛ?ÐO¼¿J¯T¿.>x?Oê?h§?ñ5¿¶r?S®E>7
¿	\B¿nÂ>ÞO{¿Æ©?¬za?³²¿dF?¿x?*y¿¶é?_Õo¿ ï¡?K¿	¿I¿pc?´£ã¾	]?Æ¥®¿åÏ??vg?Z"µ¿n>Òãù¾=`¿V?ZR?Vã?À?O¿«æ¿SP?Îv¿J­?>¶¾¾½o+?³9¿ád¿~{W¿ßo=?¨®?!æ¿ØL)¿áE?õBp?,&[?¿4A?a[>ç
Ü¾¶¿QÞ=LH¿Þ$?|3?¹·¿¹Z?F?Ê:a¿ðß¹?6é>¿Jç?`¿~O¿éfi¿Ç%5?R4µ¾]ê>£¿Ø]c?¾.?µÖ¿òE®>Ú¿ù¿g:?,y?J&?³ã?}u¿±Y¿#¥?7¿®¬? ¿×¾»¾Ñ6K?¡\¿.¿§l¿`?Ï?2¯±¿®H¿·;j?ùe?ªâ?¯Í*¿e?Y:>;j¿d7¿Eß>!m¿Ç1 ?ÁT?ü§¿ ?\µj?>}¿%TÜ?fLb¿Ì?-¿Flu¿3U¿¬¹V?èÊÖ¾?Ê¤¿YCÛ¾K¿O?à(¾Ó>©X ?9»¾¦1ð¾´¿|[¿¸½ì>&Ç?d@¿ï°>z&¿P>ºR>°øÃ¾&.Ô>%]?)Rö>OØ¾§G¿Z+?Á>âá¾»R	¿-ú¾E·¤>zÚÜ¾~à³½¥{>ÉÚ°>"X~½­ä>*|¿é+Í¾Üÿ!? Òù¾áWâ¾I» ?ÞyT¿§;Ú>òY¿®] ?­ì>g?Ï¾##O>Z¾ë?}.?EL? Â÷¿þÉ>Ãæ*¿;¿÷1?F´?Ê³?rP@o£¿j´¿&¾?/GS¿DBÇ?=ôø¾ê¯¾?~j?[ã}¿<ý¿°^¿¨?pæî?§Í¿3g¿¶$?Q¤?¨à?E¿"?'<W>}¿\S¿+>qÐ¿Ú¸?nu?æ×Á¿w?åj?=	¿õ=þ?¹¿àP°?<¿s¿ ¿°Æw?Ú÷¾s ?þ'¾¿Ì¾×5¾x±>/f½PzÃ=î¨/>DË½^$¾ÆL¾£2¾²">T\N>=õY¾)©ñ=Äéc¾`=CH=n¾ü2>Ïk2>ó(>+0¾¥ ¾Àj>o>Þ¾)ò;¾=n+¾Íoá=g"¾Ì/ö¼*!¬=ßò=®¼}>0oS¾(g¾	¸]>kõ*¾&ä¾é/0>Íf¾XW>Á«I¾Í¯/>Gö!>|6>&´¾¿=Æ·½cY>t¾ò·¾ë ?w(é½Ô%F>&²>	WN¾S¦¾ Ï¾Ó?¿-8¤>å-Ñ>[ïÜ¾(öt>±ç¾sR>¿ÿÊ=Zð¾².>¯Û´>YÝª>S6¾a~
¿Ç¸í>³>>_°¾W¾¾«Å­¾d>Ý2¾y½
{.>;[u> n0½J >~RÖ¾R¾M¿à>3K­¾¿¾ù²>8c¿a>ìlÌ¾²>,¤>æ¹>¢£¾*¯>|:¾éxÜ>æ²>ëà×>ð(¿	>i¾pÑ¾Õ³r>ôùÃ>
'ô>b3?À(Á¾É
ö¾Kï?©¾Þ?`Á)¾ÒÅî½å>ª­¾åºÔ¾·ùÈ¾û®°>^æ"?ªÎ¿ç¾7M¸>à>4eÌ>¨d¾@2´>aÃ=:M¾L¾O=º¾gü>»f§>C-¿'ÕË>î¬¸>Ò¾q\-?ë²¾Ysð>GxÑ¾"Á¾L°Ù¾Áó¨>Q)¾üÏZ>£©¿ #®¾"Ò¾ñq$?¥y¾«Ýb>¥ÝË>Õ>l¾2Ã¾¾)¨í¾´P.¿.¼>
ï>«ôü¾v;>5A¿4=%>kè=¤¾¨>Ï>é Ã>û«¾ã¿v?³>ùe³¾éÚ¾õÆ¾TÑ>÷f¯¾ÍÛ½ÃÄG>Su>. J½Äµ>tbõ¾ò¢¾© ?ãhÆ¾$Ã³¾MzÌ>¶¿(¿2R­>¯ê¾åË>í÷»>©åÓ>u¤¾A$>²ýT¾mü>W¾U¿ö¿@ØÉ?ÒÔ£¾;?;z?¿ü¿À%j¿Ú¿«õÕ¿Èf?{û?>¿ ,?TU¢¿½ÑÊ>Ä£>Ç	?¿ÈÖN?*~?Çp?ËS¿æ Â¿À	§?Z¨<?â2\¿ÅÝ¿È4t¿È ?8KW¿MY/¾«3õ>g,?ñ÷½Ðë^?¿¶H¿àë?­s¿>¥\¿bûz? Ï¿G½T?\¤¿ãDz?Î·f?[?ÜI¿EìÉ>8·¿ãê??^@¿Ü!h¿È¨µ?r¾Fú>ä4a?æ|¿K»R¿yD¿À¿Ê³O?H?··¿pé?g¿R¶>`>øî+¿t':?'¿d?iX?vü=¿-*¯¿`U?~Ê)?-F¿Dõp¿ÐÈ[¿?pÃA¿)Ð¾5®Ü>[)?q%ß½Ã H?*¿M4¿æ ?ê-[¿tF¿ñáa?âiº¿åv??ôF¿²=a?&¥O?!j?7¬5¿Ìºµ>zIë¾Ðl?äÉ?ÆT?ñ¿îÄ>j&¿º¿ë-?f?K®? @Ï¿å¯¿¸Ç¹?ÑûM¿ DÂ?-·ò¾¹²ª¾ôd?¤w¿¿L­¿|?ßéè?XåÇ¿¦Äa¿ÁÁ?ì2 ?-?¢'@¿kÒ?g×Q>±·¿ÏPN¿][>Ãb¿:8´?Yo?`ü¼¿2¸?.?-¿ùÞ÷?D~¿Ëå«?â¿¿¿ ¿.q?¤ñ¾¶m?d¹¿?÷²²?ØÀÄã>Hí@¿^­¿;çH?z9¢?Ê?ß<@{ä¿ðªË¿×?×n¿`ðà?÷¿z¦Å¾k[?ëM¿Â°¿Õ\¦¿A?6Ø@Kuç¿0µ¿x?J~¹?1©?¯~^¿)?\ùr>!â)¿@än¿È+>Sr¿Ù¬Ð??VÓÚ¿Lº¨?´Þ?Rã­¿@pd¿
Ç?âd­¿6Ù¿2´¿ÂÚ?üå¿® 5?Ç©Ö¿@^=Ä=1áÑ½aZª<ûÅ½½EÂ=âws=Ú¨=zÞ=ç÷o½VÕ½&l¡=!ú2½Ë¨=äÒ¼^Q¼¤F=W½$½Æ­y½[=M`Ê=ý¯­½é*D½Þöd=÷1=kí}=ö&½7Ý_=;T6<töþ¼úC3½àç <Ëg½J=ì÷O=55¤½v:}=Çme=|½^_×=?5]½-\=§½ýæo½8½)åQ=öÑ¼Në=¡½1¼Z?ù?÷Î¿G¨§>{¿	¿_?o?yB?ÐôÚ?É+l¿5j¿9Þ?%0¿Ô¦?HÏ¾ø¾ãC?[«S¿É¿Rºu¿ÓX?y,Ç?`ðª¿;A¿TWa?þ?Áèy?½Q$¿VR\?q3>íú¾Åm0¿Y»ý= d¿ï?p­L? ¡¿¡8y?[Ìa?îk¿äöÓ?!µY¿ÿ?¿$l¿Ò¿ßN?t£Î¾±Ä?¿8A?æh?®B¶¿uï>qû¾®óa¿rë?ÓmS?®³?.3Á?ÀcP¿ ¸¿.?­l¿,?ö#·¾ÆÌ¾¡,?)Å:¿ñe¿~ÒX¿j>?¾¯?¼Ô¿VZ*¿lÕF?gÁq?\?pý¿gB?ÛU>*iÝ¾Ï¬¿}âß=»JI¿ýû?Ì4?O¿ç[?°<G?O¡b¿Ð»?@¿y´?üa¿UP¿pÚj¿ F6?ÂT¶¾Ïì>ïâ¿+ÉX?sÌ?¹·Ì¿É)¦>q6¿âÊ}¿?îzm?ôí?JÙ?ýj¿¿Ès?<.¿Õ¤¤?Ã´Í¾«¾àÁA?tÈQ¿ä¿¸s¿ûV?fÅ?fj©¿ÇW?¿<U_?Å?®w?ÜÚ"¿²[Z?;Ø1> ±ø¾DÛ.¿}xû=b¿X½?}ÚJ?N+ ¿ w?9É_?ç~¿QÒ?sÄW¿Â¯?ÎÔ}¿} j¿5å¿»L?ÌÌ¾?_¿³ä2?CßW?Dï¨¿>Aé¾|nQ¿ù±ò>tøC?+%t?³?E'A¿æv¿Lî?¿Ý?À©¾ýÃn¾Ûã?V-¿D¹T¿-øH¿¡­0?å¢?Í¿Ìå¿ÎK8?g`?£cL? c¿Þ04?AÂ>ö8Í¾ K¿üÏ=:¿x|?re'?@,¿ÓK?«8?jR¿[­?2¿qp?¬vQ¿§A¿¡®Y¿vò(? ©¾OÎÚ>¤¨¿õ7¿Ë]¿íX­?m³¾¸%ï>òæV?ßù¾çI¿Âz¿zÀ·¿â2F? v|?%S¿ãÒ?j¿1/®> u>
$¿ó¢1?¾GZ?8N?K5¿e&§¿tt?¨"?0=¿Òîe¿aºQ¿Hæ	?Íå8¿ ¾HÒ>â?¢ïÔ½¸r??U¿ÒÄ+¿ ?&Q¿f~=¿W?Wâ±¿0´6?X¹v¿ZïV?é$F?D^_?4\-¿j­>uà¾«?½{¿/ã¿y¹í?1ôÀ¾û#?[?Â*¿zâ¿ÃÇ«¿3þû¿ç?­?¥Ö¶¿Ø¸J?0¿¿Oßî>Éþ§><ÿ`¿Fs?¬?äf?x¿½9å¿»Ä?{1^?««¿©¿ÝÎ¿¸=?v}¿ËN¾õd?~K?ü¾F?±]±¿Kk¿Oþ¹?i¿ï¿UÌ?5òó¿Gz?þ,©¿Ý`?yÝ?5)?ø½m¿Ñí>ó¿ t¶?¿Ñ@¿=­h¿Ú¶?Ë¾¿3û>¼a?>Ë¿Ò9S¿J¿ªÁ¿0P?û?¿sF?q¿ëö¶>­>3V,¿9:?He?*X?n>¿X¯¿£¯?o0*?¤F¿ñq¿ÆL\¿ÇÙ?Æ7B¿ê.¾µ2Ý>?l«ß½9I?Ú¿am4¿<v?±[¿¯G¿ib?ÏÙº¿Úé??¿ïÄa?Ò!P?¬ j?L6¿é'¶>¿Öë¾À?M¿úx¿(Â?C¾qí?
³p?ðv¿:a¿>L¿LÏÍ¿ªý]?8b?NT¿%?`&¿ÿÃ>Ú4>ëÂ7¿×õF?¥{t?Nùf?VK¿7»¿ë¬ ?´x5?¬ÏS¿hÄ¿xçj¿t?ÕO¿n«(¾¥Üë>îÕ%?Nî½nV?Ü¿ec@¿òç?ëAj¿¬=T¿ÿkq?Ø<Ç¿Î¢L?¬+¿s¼p?î]?.z?¯+B¿D;Â>'yû¾@?Õ/.?W1R?y}¤¿ >íâ¾ðëK¿eOì>Ð>?Ò¸m?í\®?\<¿Ôo¿f}?KE¿{J?ÊH¥¾á{h¾û®?d(¿ O¿ ®C¿®,?? ¿W¾¿r3?4/Z?G?Ú¿Cs/?Ñå>ÄÒÇ¾+¿XÊ=ª5¿¨su?þ"?²¿ÌvF?¾Ï3?£L¿Ë¨?Y^-¿j?éóK¿<¿ôS¿$?É¤¾ Õ>À~|¿é=ñj¦=³;¾®iÓ<ïª3½âs¡½;=N=´6¼=É
>Wç½«½½gTÈ=^½ÏzÑ=­Ü½¸¼wv=¡t½ý£½Ûí½ì3='û=×½ss½T=º¾¬=Á=4O½(é=|Fb<65½4y^½Àù< Õ½mUÂ==TÉË½¶!=]=òï¡½A¤>:C½%\¹=2z¡½ØÜ½'Ð§½*>=H½í­(=éÇ½
 
Const_5Const*
_output_shapes
:3*
dtype0*ä
valueÚB×3"Ì@%M¿w¿±¹Á?¸=¾X¡?F*p?²'¿º`¿ü¿[ZÍ¿]?ã?uÿ¿3%?§Í¿%©Â>äæ>Z7¿ËF?»ðs?vf?öJ¿§Ìº¿ Q ?5?RWS¿>{¿ÿaj¿E?*¢N¿K(¾¡Vë>´w%?Ë÷í½7ôU?Ì¿ö?¿¢?Ð¼i¿ÅS¿Òâp?£ËÆ¿.L?*Ý¿ª3p?ëo]?_ y?[½A¿èÌÁ>Dêú¾¯?
Ü3
Const_6Const*
_output_shapes

: 3*
dtype0*3
value3B3 3"3&á?¥¼Ä¾(ºy¿+d¿ä\?>E¿PÑ¿©$¿ê¾ôy?2?áL¿¤7¿°y¾ÞK&?|³D?m»¿ìwc?
È¦?ð:¿]ê>"?4¿so?t°Ê>Üéå¾²N?A®5¾<Á@¿ÔÂÎ>QÛ¿¿	¿HòÃ>ÿB??²b¾±°m¿Kb¡¿-a¿½ÃA=Àë>ÇL¿ïb?)Y¿1?êÇ»>$ÒC?|¿ê:¿°IR¿W³z?ý¤¾`ÛP¿A'?¿ÈÅ?¦ö$¿ïÐÚ¾­¶	¿EÂz¾Hd?¤â>(«*¿ï(¿_ÓP¾?S$?.ºi¿¶=>?l|?XX¿TÄÃ>pt?öu¿l?i©>5IÀ¾Þ,?hò¾j5!¿1ì¬>±>c¿@Y¿`¿¾à£>T\?8°ì>°=¾8ÊF¿Ðø¿WS<¿"=ÿÄ>+¿w=?ýþ5¿I{?|>ÝÅ#?99S¿ ¿Jß/¿îì¿ÖK:>pyì>nØ>¿ÝÆº>!Àw>sì>bõ>Ó;¿`N¾<Á>Ä>`pì=x¾)Cº¾Q?e×¾2î¿Ò±>H§]¾Ca
¿NÄ
?ôí¿Çî?¾R¶Y>`ºÃ¾ÿ	¬=¶>ÖÉC¾¥ ?Hö>¯Y?69¾æÜ¾5þ¾ÃªÖ=á>ÇÑ?e:Õ>P{·¼²_¾?ªÁ>1Ö¾ÇÎ>¨¾ÇÐ1¾Çm¹¾g'ï>Ð °>È Ç>GÎ.¿»te>'¡?I?^6¿Cæ>â>À>°Ø.>r,¿ü¾ î>U²¹>>òóÁ¾jå¾ø"?4¦¿ÅB¿}Ú>¾Tp*¿Qê*?õ$¿÷el¾8>§ñ¾SåÓ=ÉÏà>²%q¾ls?«?&3?·d¾ÿ¸¾	¥¾#3>/
?:9<?IP?Hýá¼\¾,î> ¿ìÌý>2Ï¾³[¾:cä¾G?PÙ>µBõ>xF?µáÇ¾¸}¿t6h¿¾Ü?µeH¿¢è¿PK'¿O¾m¨?©©	?ïSO¿,Ã!¿W®}¾^ô(?gØG?9÷¿¾g?r©?í=¿8Ñí>x?àâ¿7²?àíÍ>¯é¾S R?µ8¾ÖC¿êÒ>A¿+¿(ó¿Ç>Ê ?¨Ã?FRf¾W}q¿°ö£¿Çd¿¤ÜD= Oï>¯ÉO¿÷)f?¤]¿	`4?TÈ¾>uóF?L¿2=¿5¦U¿d8ÀÉ¾6?÷ç?ñMÔ?>Àx7·?5s?ó?¯@?pý¿¸{¿½?àä?¦îç>8x¿G¶¶¿kË@JÓ¿ëÀ
¥­?´mY¿¾À.@n`À:F<¿úU?Mÿ¿¿Â¨>à³?x@¿¹cü?Óñ?o@Ñ6¿¡¿`p¿FÒ>9ÉÜ?è@í)Ñ?ÿû³½SËZ¿>ù½?knÒ¿H"Ê?<é¤¿m.¿öäµ¿mê?ðB­?UÃ?Ùõ*¿h`>m?qZ?	[2¿»üà>§7>zÒ»> +>¬¿¾ªÄè>µ>g>°¯½¾^à¾Àb?.»¿S=>¿¤;Õ>Ú¾}°&¿Ë''?T!¿¬2g¾+ >Åë¾<Ï=¾ÝÛ>¨×k¾÷?xU?÷/?µ_¾´¾®g¡¾¥J>?;8?Èl ?Ý¼V¾ÝHé>4¿ 7ø>Ê¾S1V¾]ß¾.
?-ÃÔ>Ýï>À¼ýº¿;L_s<5¾^<Bb¼9@<1úþ;Íx <s;ß ¼¼{ßF<i*<øUs;"¼²?¼#-<®]¼¢¼².6<wä»j¼Ð<Ö¼Å»à;öoI¼°1;Ù;<ßÉ»Ef<²w}<<Çõ¾»þ;¼Êæ	¼Æí\;?¤g<óF<©r[<sÕ<ºFå»nPG<Ç\¼T<ö-¼ ·»uÖ>¼E!v<ÆÇ5<ÂïL<³¦æ>Ya¾h'À¾½Ý¯¾æ ð>RÅ¾QI¾{f}¾p´æ½eÒ>P>¾uu¾ À½Pê>MZ>
	×¾ß¯>³T ?v×¾[4>ãà>á¾
¨Ù>Ûõ>è0¾Z>Ë½öP¾ê>DÑ¾êÈ¾7ì¾¤Å>øs>tÂY>o®½[ä¶¾[ø¾¸C­¾Þ<ù=5>5^¾P®>óp§¾e>2}>é¬>ÎTÂ¾3¾ Î¡¾'Âê>o¾^Ã¾rÿ²¾Õéô>0y¾ÌæL¾Òô¾#Ðê½ÒÃÕ>;T>ÓÐ¾kby¾ßÃ½w<>C>NÝÚ¾Á$²>¶?/g¾jQ7>Èää>å¾AÝ>Ò¼>õ4¾`à¡>ØH½õ¾*í!>XËÔ¾5°Ë¾clð¾ôt>9ãw>$£]>9±½&º¾-Çü¾Y°¾¿<0x8>+ ¾'k±>@lª¾"
>Þ>É[>±ÊÅ¾y¾>°¤¾×ß?!Ì¾Û¿³m¿ø¢?@"M¿Ò¿?+¿é¾gï?ê?i:T¿ò%¿¤Ö¾©ò,?L?7R¿l?Ìs­?±jB¿.pó>öú?½g¿­?äËÒ>ï¾úöV?ò<¾óvH¿õ×>kJ¿>¿Ã¢¿­ÈË>$?)?áÃk¾2w¿ÈÖ§¿N/j¿ËI=÷ô>ò²T¿k?aPb¿n£8?²JÃ>@§K?OT¿ÛüA¿î²Z¿e?f¯¬¾|2[¿üH¿:??q!-¿E¦å¾¿;¾Yo?%Ýí>B3¿¤À¿*[¾Q÷?\§,?Lu¿á¨G?Od?è$¿uÍ>
E?Ù ¿ñIx?ëè±>JÎÉ¾m5?9x¾0)¿Ü{µ>Ü~n¿+Jd¿»¿Çý«>çé
?hø>¯ûF¾½¡P¿|§¿;¦E¿*=è¿Î>ý3¿ÜØF?¨?¿.Õ?öÒ¤>á+?T®]¿5¹#¿X8¿ÆM¿(«B>÷>
(â>¸¿·+Ã>=q>Wî¢>V>¦
¿¾«ëÉ>">÷=O¤¾¢Â¾\C
?»á¾-%¿hù¸>cg¾P¿Î ?Íò¿óH¾c>6Ì¾JÅ³=wº¾>^L¾®m?­ ?íá?ìâA¾¾É¾}Pà=ü0ë>ô¯?ÏÞ>Aº¿¼Ñi¾Y^Ê>;)à¾R×>¹«¯¾Î9¾ÃÁ¾Ùæù>è¸>¼Ð>ì ¿qXÈ>ÇN~?dÀh?;¿ÀÜH?7?±®'?ª>Ëú¿pû	¿ÏO?C#"?E~>»X)¿OH¿K?¤g¿.×©¿Q^>?}^î¾ÎÐ¿Q;?¿4hÎ¾q!ê>}R¿[9>VJD?³Ò¾?Y?§m?ËO?^Ç¾!*!¿¿Ûf>Êr?X¤?ñNe?QE½ÈÝï¾EP?°²f¿ù]?.Ë4¿¨9¿¾¤iG¿<?Åò=?%V?K'u?ýå ¾X<L¿ì:¿Â?>P!¿~ùÕ¾ ª¿å5u¾ý:_?Õ Ý>pä&¿­6¿4L¾È ?Þ ?Gd¿!:?Uf?Ââ¿ro¿>ro?²o¿xWg?7Ä¥>
¼¾Z)?»¾H¤¿´©>7^¿&µT¿¾{¿}@ >n?sç>Áf9¾6dB¿Jü¿(8¿²w=E£À>:C'¿OF9?ø1¿=2?é>4& ?ÊN¿c¿û+¿"]Å¿?&l¤?>|?¶æÍ¿Þ?IC,?oÔX?ãhÅ>á¶³¿®l2¿Ô[?ú¨Q?Ùe¤>XûZ¿x¿i ¸?bÄ¿ÛÛ¿D*v?
¿ënÀ¿¥øÀ?>º¿Ês¿`?Y¿z=o>¡Ò}?"¿ûå²?,>«?\ Ê?D¿ÒfP¿-U:¿xB>I?YÔ?WB?'½Ü¿#¨?Y(¿¸F?µÈi¿åE÷¾î¿I¦?2u?t? ´¿!G>:Äü>IWç>D¿¥Ç>æh>ª¦>)½>+#
¿]%¾¯Î>¶'¡>ºü=ÿQ¨¾WÇ¾Ên?¤<æ¾­Ï(¿ø6½>«ìl¾ìé¿ÊS?(¿ø'M¾2¶h>6Ñ¾Pä·=ÌÃ>GQ¾	?1 ?H]?ÊTF¾0 ¾~9¾ítå=Bð>Y#?.ëã>sÄ¼£in¾þÏ>ÄLå¾*BÜ>¯²³¾ú>¾B4Æ¾s¡ÿ>Ì¼>áØÔ>õwÒ¿T"
?W¯?7z ?¨Û¿}?p³7?$:g?~Ò>±¥¿¿E>¿®G?Ð_?OP¯>¢i¿å¿8Ä?&¶¿4ê¿PA?ÁY$¿î5Í¿ÎÈÍ?cÆ¿<P¿£m!?¦ ¿ >V?,¿íÆ¾?ü¶?#×??	¿D=^¿~´F¿+>sã¦?®â?y?½b%¿?À¿'Ê?Ny¿Ø¿®}¿§S±?)÷?	¦?ö}a?þ¾Û;¿µî+¿?k?C`¿NÐÄ¾{»÷¾ea¾ÄSM?ÚË>ê¿oï¾éÓ;¾é0ú>£÷?¸9R¿¥+?¹ëz?Ï¿°>ÛÛ[?7y\¿¶ÉT?Îx>ó¬¾¥|?«¾¯ÿ¿í>eL¿ú¥C¿ôîf¿Mf>^î>ãÔ>7*¾Í2¿æÌr¿c)¿$Â=(0±>Ù¿^j*?)²#¿*?ÕA> N?
ü=¿]P¿0¿¾¿&e©>aW?ËD?Ó¡¿Õ)?/Fá>)Ç?n>Ïk¿þTé¾¢´/?	?#ýV>c/¿C])¿ p?ÛC¿M¿õ ?oÉ¾Ø¦{¿õZ|?ñs¿0®¾öÅ>£ø1¿nn>j÷%?²²¾ ói?¶ð_?õ)?é¶¨¾_D¿¬ó¾+1C>)¨L?ô?9âA?þÕ&½ÏÊ¾l0?C¿^;? Ý¿¯¡¾=(¿uY?ª ?5?ï®?ä¾i¿@½¿Æµ?úe¿Ûò¿B?¿M!®¾?<b?_m¿ï8¿Ú¾(A?myd?M¢¿?¿¸Á?Ï"Y¿gñ?½©?7ª¿H¤?në>¿Îp?S¾þã_¿Ä(ð>;Í¿¿aJ²¿ã>fÓ7?þ[$?{¨¾Í
¿Îs»¿Æ¿@a=ÿË?ým¿p?¹Â|¿7N?ýÚ>¯sc?­¿#¨X¿wAt¿7>+?Ç`¾S©¿¿¦2?ø[á¾Òv¾ü!¼¾jH+¾òí?rÏ>3'é¾réµ¾Û£¾üÿ½>½à>9¦¿ò?Û>?èÕ¾]¸>÷&?n'¿e!?g>­W¾×(ì>¾Ï½Ñ:Ü¾~;l>²8¿C¿`/¿Rà_>îÑ´>¬¡>a¾úÈ¿(c8¿&£ ¿*bÝ<b>«é¾¶j?³ ø¾Ô×Ê>ÿV>»ß>'G¿=Õ¾Cð¾bx¿â­>[¸\?ÏJ?W3¿`U.?»>ç>*?F>z?q¿8ï¾Ø\4?6¹?æ¯\>ðú¿sÚ-¿Í w? I¿¯h¿Á9%?øâÎ¾0)¿¡?z¿[%³¾;5Ë>9°6¿Ü >]*?§¾¶¾'p?6àe?ºª?°/­¾ûà¿æ!ú¾]H>ÑR?o£?ÇG?B+½¤/Ð¾HÃ4?:H¿cU@?Zê¿ ø¥¾H-¿8_?iÜ$?¤Ü9?MÎ¿.$Ï>w?¥p?Ò¡¤¿ù¬O?f¼	?Ê^-?³×>¦±¿¡©¿¤ÛV?7£'?r>H/¿O¿3?o¿¯¿kÓD?itö¾Ý¿3K?Hê¿hÕ¾ò>ã Y¿ôI?>ÜòJ?²Ù¾
?ë?(¡?OÎ¾¢¡&¿[ü¿Å¯n>­Bz?3ë©?/m?	L½µ ø¾«UW?ÿn¿Je?#í:¿3¶Å¾G-N¿éô?9dD?°h]?x³¿¤ë>?½à?kH»¿¢?l?«¯?)9E?)³>Åv£¿nJ"¿Ekt?¬³>?K>B.G¿k¿]§?9¿üÂÇ¿îç_?w.¿c¯¿©¯?Ng©¿(Åò¾]°	?w¿qY>ßf?¨¥÷¾Ã¸¢?*Â?sÙ·?¤±ê¾¦=¿Ý{)¿WÃ>X? LÁ?_Ú?¦h½ß¿öt?«¿R?ø¤T¿ééà¾%j¿Ú??oi_?ß{?ð¿ÎÊ>I·?ek?ä/¡¿vTK?Ú?¾)?$>â¯¿d­¿¦\R?0!$?[²>Sm+¿ÅJ¿?|j¿Tí«¿µ@?)Lñ¾Ô¤¿¦?Ì¿ZñÐ¾Èí>U¿8I;>«³F?ã#Õ¾Z?#?e;?ìýÉ¾þ$#¿>Þ¿!±i>	u?ó\¦? h?&¾G½)Ðò¾ ÔR?;i¿èR`?Ç7¿Á¾ÊÜI¿ª,?(H@?ÆX?/cW?È\¾p3¿Y:$¿[´`?#º¿þ»¾¡ì¾pW¾P D?ö·Â>ì ¿~Îä¾5i3¾Êúî>3V?ÎH¿³q#?<­o?R¿1¨>³R?R¿¬@K?±£>u3¥¾ò?D¾M
¿­>W<C¿á:¿Á\¿`Ë>ënã>WYË>ìã"¾ñÉ*¿ëg¿gÌ!¿:=?©>3ô¿jÇ"?J\¿4"ÿ>_í>H´?x5¿¹¿j¿QW?¯·Â¾õ)w¿6b¿Â?F8C¿y¿Óø"¿'`¾W?(?©øI¿F¿| w¾æ$?®B?BL¿6"a?Ë¥?O9¿J¬ç>¢?#
¿Îû?ÛÈ>¹ã¾_L?ÜÐ3¾½Æ>¿£Ì>Tv¿Pµ¿³ë¿fïÁ>#£?Ë?ì^`¾@k¿;º¿çÝ^¿Æ?=Ñ é>_kJ¿§7`?[`W¿·/?}Ú¹>ÏA?îöy¿È8¿ !P¿ïv¿ ¢>¸M?H<?ËÎ¿'|"?O×>þ¤?Éýv>Ú`¿á<ß¾¹(?Ä(?,°M>¢ý¿	"¿47f?þa;¿ìc¿ ÿ?\ÓÀ¾×Ãp¿*pq?i¿gø¦¾e½>£E*¿ú©>]É?Sª¾¦Ô_?@V?ä|?mj¡¾<_¿Û!é¾s¿:>ÍC?­ñ?þ~9?P½k	Â¾2z(?Ä:¿ûB3?/@¿o°¾óO!¿ÎP? ¨?×:-?ÉÖ¿iÏ?£¼²?£?£Óß¿f,?B;?ì´k?Ö>-\Ã¿;õA¿I?¬éc?Éµ²>Ìn¿ÛÈ¿BÈ??Î¢¿½î¿GÌ?Ü'¿®/Ñ¿gÅÑ?huÊ¿¿?$?kð¿·>Êõ?ü¿yÂ?ª&º?(¹Û?>¿yb¿ûJ¿A¢>$ª?¢ç?*¡?Ï®½@(¿=a? $¢¿ë¿?#~¿pf¿'¿Ã´?°?N?m81¾! h=!¤>_ >ðâ8¾Á9é=j®=³Â=ûB1=P_!¾Ð6 ½¡Jñ=ýB¼=x=Í¡Ä½Mè½Ô8%>G{¾ 4E¾{
Ý=Fc½Ë,¾´F->d<'¾ê©o½Ví="gô½ ÒÖ<Ãêã=pzt½¼£ >óÃ>65>Û°g½·!»½°P§½=V>ÒÒ>>¢ >lå»ÌA½«Óñ=*ï¾4§ >iìÑ½)	^½Úç½gP>Ü=ÿ¥ø=&X¿½Ç>Ù÷3?¶$?^^a¿^%?Á¼>Tí>X>´´D¿IKÃ¾Ý?{å>óð3>¯ï¾#Á¿fI?]í#¿bp¿=¸?P°¨¾ R¿X7S?tÚK¿ã¾s°¥>Qõ¿	î>é
?¾ÐC?ôn;?¦<]?ç5¾ÿä¾2óË¾*_#>)K+?
h?ÒF"?b£½¿©¾cc?B#¿Ò?=ãÿ¾vS¾¾¿æ6?!l?¼?
 
Const_7Const*
_output_shapes
:3*
dtype0*ä
valueÚB×3"ÌsZ¨?|üÜ¾A¿]¿Å¢¯?l]¿#ñ¿Aõ8¿zd¨¾UL?«2?û7e¿×2¿§;¾0Ë:?2ò\?ô¿x?ªV»?8ûQ¿²v?Ä%¤?@¤¿NÞ?¬ã>G ¿¢,h?L¾mX¿ù>è>$¿g¿vj¬¿.Ü>ÊÄ1?ñ?Õ£~¾c~¿Fµ¿ßî|¿Ë¥Y=J?*ºe¿Dw~?¢nt¿kG?(íÒ>õ[?à×¿Q¿5l¿
Ü3
Const_8Const*
_output_shapes

: 3*
dtype0*3
value3B3 3"3Ò(?÷zÉ¾N}¶>MÑ>åQþ>¿UÞ¿H?bB*?ÉÏ¿q¾y¸¿ÝTÒ>õ¹4¿h1¿v0ø¾BJÝ>v¿¼v¾äâ¾ëzß>Åäõ>Åj¤>Ke¾ë[½«"?¾*æ¿~>[×	?'Ç?F³>I_¾öÔÅ¾úé¾hØ> àË¾õ¾*Ùà¾Úd)¿f¯D?9s¿tu÷>¼;«¾þC\¾^>âxâ¾?ô¼½¾M¿G?<ã?âx¿õgu?î?- «?0ì²¿¤®¿ Þ@õä?P×Î¿ÄC¿9"±¿ol?ÿó¿
î¿êà¦¿½Ê?±¿Ùæ%¿¿·C?ÂU¥?V]?hU¿úÞ¾êÀÚ?Õ,¿«_±¿7Ò*?R]¹?¨jÉ?×q?JO¿ß¿yR¿¦.?¿!¥¿8/¿®Ëã¿|?@á»Ó¿,c¦?èDf¿e¿sT?½F¿E§»?_'¿ã£½¿Î?Q¬?UM¿ú9?ÂU?U?>¿ùY¿]jÌ?¸­?À¿ö[¿.=¿UZV?.¸¿Ãc´¿:ï|¿]a?¿5tû¾jf¿¾Àc?Lz?'?l!¿ à½»Ç¥?©ú¿¾k¿t?z?=¤?|³6?ÝR¿@I¿4sn¿ðS?ÆO¿Iz¿¯%e¿ó¡¬¿rÈ?Ýu ¿¤0|?Ö.¿zà¾Vâ>ZÍf¿6? ]A¿·¿ ?ìx?¿¹?ã¥?ß:?ÛC¿Ú>¿Ìb?B6z?d
b¿ûïÕ¾aA¿?Ì¿$¿b^6¿l"?B¿Mµ¾ô!&¿d6$?®4?Q ñ>ùÆè¾Æ¡½o?üß¼¾ÖA¿^­º>J?×\?Èº?HÝâ¾À]¿Ûì+¿âÑ¾>øÎ¿pu4¿Á7%¿²ðx¿
?:cg¿øÔ5?´¤û¾¶Ù¡¾ê0£>9i&¿gM??k¿;>O¿³¾a?1t¿jÉ®>ÝO¾9fµ¾3 Ü¾ÀØæ>=Sá>.¿¿³¿ o?÷|>âä>w¶¾OÈ?ö?²N×>Ôø¿¾ýå>V>#Ä>4ßÁ¾ÞPÕ¾H¢¾i>¥È>=N¿#ý^>)Ùä>æd\¾w(ï¾-ï¿©¾}ë>'«>JúÊ>öHa¾ÅÝ°>qÕ>Ã>ó?| *¿?w¬Ö¾>O?>ª@¾°wÄ>vò¾©¤>¯¬ô>QB¿\ª»>ø_¾BÜJ>Åqh>ÃZ>.ç¾]¾øÞ>ÖC½>7ûª¾Ó!¾l¾rÏi>æÈ¾¤ÃÄ¾sò¾þu>ù¿¾½#	¾åT{¾?mx>Í«>EÅ6>¸0¾9xô¼Ô´>XÞ¾h¾Ä4>g:>P¦>IG>½+¾Yê[¾%¾öV>Æ¢b¾¾òy¾M¼¾¤Ú>¡¯¾>Y>¾vÚô½¬áö=·À{¾Å>ëR¾5Ã¾öÁª>ä1¾øVÓ=hk¿½ËUÛ½ëa¾Ë>;9>ZeR¾F2¾¡V!>³=*
>¿Ü½ø=>Ïª9>Í*>mè½<y
>Àg=;(í=jê½ö ¾yv¬½%¦=¡®f<[¡*¾¤Ï=Z
>>½"¾w¾Ø¼½'í¡=CÏ=mõ=3½©ÚÕ=ÐÍ >èÙë=æ®1>UON¾¦'%>¹È¾Ó³=Tg=Bõh½øí=._¾ÜÇ=éë> !¾KX¿S1?äê¾¿#¿	¡*??&? ¿°XZ¿ß@E?±º>Mì(?BÞ¿Åg?{ÿb?®$? å¿xL)?56>ù?¡L¿×«¿©ÚÒ¾Ñ!Ë>J=P¿)Ò¤>æ&)?2ç¢¾ºÅ0¿@¿èå¾çøÅ>Ú´ý>£?¦¦¾Ðº?z?7-?<Y?`<|¿aëI?Ä¬¿vÛ>õ<>th¾È7?ô2¿£Só>Ù4?ÑþD¿S7¿Â!Û>,zÆ¾lã¾ãL
¿µ???<'Z¿ô,9¿xI'?iT>²B?+Âä¾HD?H@?l÷?X­ð¾A?-->´æõ>ó¾Õ·¿OÒ²¾ÉE¬>ý/o=æë0¿È>dt?·'¾Úê¿]æ"¿ÅúÂ¾å§>î)×>yþ>8¾.½Ý>?ô><8?¥êU¿>+?º¿&<º>o>q¾1Pö>ÀÄ¿d\Î>`?s'¿?þ¿G44?8#¿;¿Çvc¿n n?Oh?Bf³¿µG¿¾?	4??k?æ<¿Q¤¡?[P?û]?ìE¿c%l?q®Ü>Õ7J?áG¿rí[¿¿T«?p²Ä=~¿wæå>ûðk?¤9ã¾v¿'ö¿«W ¿
?èð0?ÖDQ?Dè¾#Y6?î§[?ÃI??Xê¯¿ÒÒ?ÑS]¿¹&?zÅ>;£Æ¾J?y¿³³)?B|?­c¿%Ê^?ùñ¿Ôð>ù	?Ï'?Ì/¿¨b+¿ûY?C°`?ûJ¿@À¾_Ô-¿É
?sn¿ci¿Ä#¿:?U7.¿¢Î¢¾/¿,v?@@"?úØ>BÑ¾Ü½R¬V?¼©¾¬.¿¢§>*è5?Ù¨E? ì>ú¸Ë¾­¿9c¿×Z«>ä¿÷"¿G]¿è_¿ùÇ?ÎÈO¿£H#?Sùá¾,W¾^>o¿/'8?õdú¾Q:¿¡·J?1Ëå¾ð>Qfx¾þO¾.­¾ïµ>ÿÅ°>+¿Àç¾=]Ñ>k'F>bK³>&¾ôÿõ>^ïð>ê¨>d¾t±³>ïì'>Cà>÷¾Z§¾ÎÌ_¾W>À¬=lÝ¾Åð.>³>¨ç,¾. »¾¥ßË¾²t¾ R>D¤>Ã=>ï½0¾Á>-%§>W>æ>yÜ¿QÖ>Åj¨¾ i>åè>È&¾F">Gñ½¾">ô¿>"Ñ¾¨¿+I?j56¿íÇP¿µí}¿Ø?+«?EÈ¿Oÿ©¿1?Z??R¿Âr´?­»°?°Îw?ó\¿tÏ?[[ö>Ò¾a?á"_¿çu¿ *$¿å&?üÛ=uk¢¿úR ?4²?u©ý¾¡¿%¿hÿ2¿y"?E?Õi?A¥¿OK?L6u?`?"©?êaÄ¿25?øw¿Gø*?8íÛ>¿Ý¾ªb?T¿5r=?¿Í?Ä_¿÷¢¿
÷A?¹®/¿MI¿hÕt¿?qz?IÁ¿è£¿?W%?Z}?j|J¿5ü­?0gª?în?+	U¿¼-~?zí>ï¨Y?ê$W¿¾¸l¿ÂH¿Ç|?·Ó=6¿·t÷>Tõ}?ô¾³¿í0¿',¿1?ßs>?Å?a?
ú¾ÎED?ëml?vX?Q£?BY½¿½?{:n¿¢Ø$?¯Ô>VÎÕ¾OZ?V¿'©6?³Â?uá¿aj·¿'æZ?0DF¿0.c¿D'¿¹??åëÙ¿ú¸¿÷§?X)?º?òd¿ÐYÄ?êNÀ?µÒ?ákp¿3m?®?Ñ£u?Ír¿v¿«¡2¿í,?íîî=Æ»°¿¢?_M?"
¿Â¿º¢¿¼ÅB¿ß·'?gïV?g4~?)¿Ý]?=i?It?æ	¸?u°Õ¿í«?m¿}	:?ðNï>dJñ¾1v?x¿B$N?c6?ä¦¿²ó¿¦?Qy¿¸õ ¿2ÄC¿ìÕL?®ïG?Vf¿>¿&Ìl?Kà>¢ÉJ?Ûç!¿â?¥@???YW*¿=K?Óí½>à	.?ì,¿¸G=¿óý¾ÌÚó>qI©=loz¿øÜÅ>úK?Ã¾å5T¿af¿ÿ	¿©í>¯H?Y4?æÇ¾ð?ä=?-?·d?¶f¿ fr?&|>¿/Ï?w©>õª¾T.?µÔV¿º?ýY?Û|l¿Dÿ¾S>÷¾`¾ØEÀ¾#.É>F^Ä>÷¤¿¸ ¿Xè>i\>3+Ç>1¾U¢?.Ò?N£»>&M§¾Ç>#:>Éîª>õ¨¾ÿæ¹¾x¾Áo>D&=Q÷õ¾	UB>JpÇ>)@¾jlÐ¾òxâ¾%¾Tki>û>ä°>RUD¾#><¬¹>Ñý©> ?
³¿»î>ì»¾u>Þ&>ýç'¾8«>AÿÒ¾|r>%;Õ>wDè¾¸¿'ú!?±µ¿æ(¿îtL¿¹íU?ÍÏP?ø@¡¿¢à¿+Ow?9ê>(ÊS?É)¿ÖJ?M?5G?%ç1¿ºBT?8\Æ>°Ã5?à©3¿½®E¿B.¿®þ>AÍ°=¶Æ¿¥Î>T?ù=Ì¾¡]¿ÓÒp¿Õ¿ç5ø>F?<?FÆÐ¾~ç#?ApE?tÃ4?.?E¿Í(}?ÝðF¿©	?M±>Ì²¾ª6?^`¿?9¾b?Züv¿þÞ?ý¿Fq?¼?Âö§?Â¾¯¿¤«¿Ûx@­äà?ô*Ë¿J@¿íü­¿|é?¸î¿áÍé¿<ê£¿J&?ú_®¿ô"¿dR¿?f¢?$-Y?9Q¿¶>¾gÞÖ?MÃ)¿H9®¿É'?¶?öÖÅ?ÑÌl?èK¿!¨¿>¿Ñ+?F¦¿Å2¢¿ä¿Àß¿@æ@GùÏ¿ºn£?
.b¿y¿­?s¿%R¸?_z¿»Eº¿ìæÊ?9=3½'"=fÍ9=ûa=.sl½[Ëf½Ò:²=I= ¬½·Z½øj½èä:= ½H½\½²¡D=<j½>Û¼OæH½áF=]~Z==Þ¾½!jÃ»/=½fä¼,gj½a¾á<öt==
L=£+	½É/½çO½ÔÀæ<Ç(5½M9Z½ËG½°½¹Ä®=Æç½fâ[='½¨¸Ã¼¯WÅ<~<I½âüw=s(½	z½\~=Hu?Å-1¿z ?¿á7?&¥_?j¿¦hd¿5c°?6¹?yB¿m ¿ªg¿]ö8?í¿î§¿AZ¿cB?q.h¿öùØ¾µÒF¿lD?4<X?þ?pJ¿eÁ½´?]
â¾êúg¿	iß>Wnr?d¶?¦?«À¿kø-¿hÁM¿;^ä>iI3¿Û÷W¿lºE¿fö¿Dö¬?u¿Y?|¿Á²Á¾~MÃ> (G¿®lu?SÚ&¿âx¿.?EÎ¿X +?s¿2¿ò¢X¿/¬b?:@]?!Üª¿¿Y?Îø> h`?~*3¿ò?,Ç?jS?5<¿Áç`?B-Ò>¤@?Ê]>¿ruQ¿¿ùì?U»=¿ñôÚ>Øµ`?·hØ¾mÕj¿q+¿Ãµ¿?¼(?¹NG?#6Ý¾«-?>3Q?$??WK?«§¿Å?ÂÊR¿sÜ?Í »>«.½¾Bê@?À»m¿¾!?@p?yÙ¿e§¿ôóG?&5¿O¿Wd|¿Ð
?Lâ?[Ç¿ö÷¨¿H¥?ìx?×¸?¬¼P¿8[³?å©¯?ÍNv?Í[¿C?·Ýô>a`?6É]¿t¿°+#¿å1?Ò@Ú=Ùo¡¿`ÿ>0æ? ü¾ÙË¿z¤¿ê1¿²3?UD?í3h?kÜ ¿öTJ?oºs?Í$_?¨?±1Ã¿©A?6u¿lï)?Ú>hÜ¾^Á`?E|¿ºL<?ó?+r¿,¿R¡>á¾ól§¾]¡Ë¾ZÕ>º÷Ï> ¿ÿR¿BOö>i> ïÒ>Ðh¨¾~´?Ì¹?½¸Æ>/±¾gÓ>öE>µ>÷ï²¾/âÄ¾{¥¾{¦}>N0=c?¿³ÏM>,8Ó>¡jK¾*¼Ü¾¡Ùï¾²¾5w>³f>zW»>=îO¾ä=£>ó£Ä>g´>¡?¦{¿Ö"ü>#Æ¾¥>]0>
Ó1¾CUµ>ìuß¾¬ë>Óá>Çüõ¾ùmð?´x¿Çò?9æ?µ?Ñ|½¿ô¸¿­Ô@zò?õÛ¿OSO¿Ï»¿8Æ?a± À*ü¿z»°¿è?¼¿¦²/¿ÿ ¿$#?þ¯?(j?<a¿8¾¢«ç?	7¿âØ»¿`è4?ROÄ?NOÕ?Q?TÚ[¿ß¿¦¿ì8?®-¿¥á®¿ ¿?ñ¿@|<à¿P6°?²Ýs¿&Ù¿¿%?D¡¿ß»Æ?)¿ÖÈ¿ÄÚ?('¿Ùu¨>ô¾cÕ®¾ä£Ô¾½}Þ>c+Ù>Aµ'¿%[¿ ?os>ADÜ>eÜ¯¾?"ÿ?«Ï>¹¾§ÁÜ>¶LN>"
½>ÌÚº¾KÍ¾¥x¾Ëo>Úà7=µ¿ïêV>ªÜ>½jT¾pæ¾lvú¾ä¾>êh¥>¡Ã>{!Y¾ðvª>NWÍ>¥ÿ»>ë¡?ws$¿f¥?OçÎ¾¡+>½*8>D±9¾;[½>Yé¾{¤>Ñë>o ¿÷>ò½¾úÐ>ÔT>}º>³ Ã¾åu¾¾
?A²ù>#á¾U¾A-Á¾};>¿Ë¿+þµ¾£D¢>;Á¾uí4¾NÊ¥¾Âß£>9O´>× q>*Lh¾C!½òî>W|<¾DpÁ¾áJ:>1'Ê>¶¨Û>Hu>eb¾¾'«¾6m>>î¾;´¾à¤¾\mø¾Ë9?'éæ¾
uµ>ñ{¾R!¾ÑÚ">m¦¾6¦Ì>±!¾åÐÎ¾Gá>úF?xí¾|×>tö>íß?eÑ¿+¿il?H¬H?I5¿«¾@¿?ç÷>U¿ÎP¿	C¿ãh?d¿Ýg¾=¿K³?³è?ÉÁ>«°º¾@½bº??ßz¾Üu¿¥·>Év"?q0?%LÓ>¬òµ¾¾+é¾Öâ	¿0
>ÞKð¾åº¿³¿-§G¿½Ñg?\9¿ÔÔ?,ÒÉ¾TÎ¾á>°v¿Xx$?¡ß¾ 6&¿Ô5?j2¿r´>gp£¾-F»¾_Åã¾ªRî>_è>>¤3¿R|¿FÁ	?a>¨ðë>æ_¼¾+Ü!??ÆGÞ>o0Æ¾úvì>±ú\>´}Ê>&È¾o9Ü¾Ö@¾GÜ>föD=K°¿æ5f>Bì>&c¾Içö¾p$¿ ¾ÌA>
.±>$Ñ>Ãh¾$¶>ÓóÛ>?`É>êµ? '0¿z?I Ý¾¤[>EE>ÜçF¾ÔÊ>Ãóù¾Uî©>=ü>%	¿á¿éB5?!-$¿m<¿cÌd¿Üeo?ü«i?®s´¿f,¿X`?÷?m?k9=¿¢?>?rH_?JG¿m?ÝùÝ>gK?¯I¿»7]¿àê¿?ÖÙÅ=X¿»?ç>RSm?ãä¾dx¿V¿¿xH!¿rá
?£ú1?R?T¡é¾ýj7?Ïñ\?ÈHJ?e?ò°¿O¦?5 ^¿º?X)Æ>ÍÇ¾È¾K?d{¿²*?ì¼}?2¿jåÂ¿Åh?0­R¿fq¿Í¿	?í?Ùç¿«Ä¿µ±?¶(?â?¨Ñr¿¤Ð?rXÌ?C?_x¿tg?ml?÷?Ðÿ¿ßï¿Ð=¿yÜ6?ãý=¼Ë»¿_?¢E?¥¿É!¿é¬¿ÀöN¿`72?cd?á?«æ¿7^k?Ã?üÉ?ëÃ?ã¿Åµ?(×¿®E?Iþ>h2 ¿ó¹?Ð¡¿t[?sÍ¢??V±¿Ð×¿ð75?1#$¿
<¿¾d¿_Wo?×i?Âh´¿!#¿øW?¦ï?Âòl?÷-=¿;¢?y4?î:_?=	G¿§ym?mìÝ>6[K?I¿W*]¿ìá¿vw?ÝÍÅ=«O¿»1ç>ôDm?ä¾aõw¿.·¿µ>!¿
Ù
?Ýï1?`rR?0é¾â_7?oä\?<J?à[?Òç°¿¼?»^¿g?YÆ>sÁÇ¾s²K?1{¿9¨*?­}?¤)¿
 
Const_9Const*
_output_shapes
:3*
dtype0*ä
valueÚB×3"ÌÕ?NÑ:¿t5)?¢âA?ÐÏk?w¼v¿¨Õp¿¹û¹?bÞ?6¿Mû¿íDt¿MC?ó§¿¹¤¿ f¿/M?ýÏt¿ÁÇä¾®£Q¿d7O?¬ÿc?s?QÞ¿jêË½îÔ?cVî¾©t¿në>¢?ºà?:&?E#¿Fo7¿íòX¿¬Êð>N
=¿·c¿&|P¿ù¿_¶?Ýý¿2se?Å¿[<Ì¾píÍ>ýQ¿Uc?î/¿Â¿tn?

NoOpNoOp
b
Const_10Const"/device:CPU:0*
_output_shapes
: *
dtype0*ºa
value°aB­a B¦a
õ
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
layer_with_weights-4
layer-7
	layer-8

layer-9
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 

w_mu
w_sigma
b_mu
b_sigma

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api

w_mu
w_sigma
b_mu
b_sigma

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api

!w_mu
"w_sigma
#b_mu
$b_sigma

!kernel
#bias
%trainable_variables
&	variables
'regularization_losses
(	keras_api

)w_mu
*w_sigma
+b_mu
,b_sigma

)kernel
+bias
-trainable_variables
.	variables
/regularization_losses
0	keras_api
R
1trainable_variables
2	variables
3regularization_losses
4	keras_api
R
5trainable_variables
6	variables
7regularization_losses
8	keras_api

9w_mu
:w_sigma
;b_mu
<b_sigma

9kernel
;bias
=trainable_variables
>	variables
?regularization_losses
@	keras_api
R
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
R
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
Ð

Ibeta_1

Jbeta_2
	Kdecay
Llearning_rate
Mitermmmmmmmm!m"m#m$m)m*m+m,m9m:m;m<mvvvvvvvv !v¡"v¢#v£$v¤)v¥*v¦+v§,v¨9v©:vª;v«<v¬

0
1
2
3
4
5
6
7
!8
"9
#10
$11
)12
*13
+14
,15
916
:17
;18
<19

0
1
2
3
4
5
6
7
!8
"9
#10
$11
)12
*13
+14
,15
916
:17
;18
<19
 
­

Nlayers
trainable_variables
	variables
Olayer_regularization_losses
regularization_losses
Player_metrics
Qmetrics
Rnon_trainable_variables
 
ZX
VARIABLE_VALUEnoisy_dense/w_mu4layer_with_weights-0/w_mu/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEnoisy_dense/w_sigma7layer_with_weights-0/w_sigma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEnoisy_dense/b_mu4layer_with_weights-0/b_mu/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEnoisy_dense/b_sigma7layer_with_weights-0/b_sigma/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

0
1
2
3
 
­

Slayers
trainable_variables
	variables
Tlayer_regularization_losses
regularization_losses
Ulayer_metrics
Vmetrics
Wnon_trainable_variables
\Z
VARIABLE_VALUEnoisy_dense_1/w_mu4layer_with_weights-1/w_mu/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEnoisy_dense_1/w_sigma7layer_with_weights-1/w_sigma/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEnoisy_dense_1/b_mu4layer_with_weights-1/b_mu/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEnoisy_dense_1/b_sigma7layer_with_weights-1/b_sigma/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

0
1
2
3
 
­

Xlayers
trainable_variables
	variables
Ylayer_regularization_losses
regularization_losses
Zlayer_metrics
[metrics
\non_trainable_variables
\Z
VARIABLE_VALUEnoisy_dense_2/w_mu4layer_with_weights-2/w_mu/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEnoisy_dense_2/w_sigma7layer_with_weights-2/w_sigma/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEnoisy_dense_2/b_mu4layer_with_weights-2/b_mu/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEnoisy_dense_2/b_sigma7layer_with_weights-2/b_sigma/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
#2
$3

!0
"1
#2
$3
 
­

]layers
%trainable_variables
&	variables
^layer_regularization_losses
'regularization_losses
_layer_metrics
`metrics
anon_trainable_variables
\Z
VARIABLE_VALUEnoisy_dense_3/w_mu4layer_with_weights-3/w_mu/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEnoisy_dense_3/w_sigma7layer_with_weights-3/w_sigma/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEnoisy_dense_3/b_mu4layer_with_weights-3/b_mu/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEnoisy_dense_3/b_sigma7layer_with_weights-3/b_sigma/.ATTRIBUTES/VARIABLE_VALUE

)0
*1
+2
,3

)0
*1
+2
,3
 
­

blayers
-trainable_variables
.	variables
clayer_regularization_losses
/regularization_losses
dlayer_metrics
emetrics
fnon_trainable_variables
 
 
 
­

glayers
1trainable_variables
2	variables
hlayer_regularization_losses
3regularization_losses
ilayer_metrics
jmetrics
knon_trainable_variables
 
 
 
­

llayers
5trainable_variables
6	variables
mlayer_regularization_losses
7regularization_losses
nlayer_metrics
ometrics
pnon_trainable_variables
`^
VARIABLE_VALUEcategorical_dense/w_mu4layer_with_weights-4/w_mu/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEcategorical_dense/w_sigma7layer_with_weights-4/w_sigma/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEcategorical_dense/b_mu4layer_with_weights-4/b_mu/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEcategorical_dense/b_sigma7layer_with_weights-4/b_sigma/.ATTRIBUTES/VARIABLE_VALUE

90
:1
;2
<3

90
:1
;2
<3
 
­

qlayers
=trainable_variables
>	variables
rlayer_regularization_losses
?regularization_losses
slayer_metrics
tmetrics
unon_trainable_variables
 
 
 
­

vlayers
Atrainable_variables
B	variables
wlayer_regularization_losses
Cregularization_losses
xlayer_metrics
ymetrics
znon_trainable_variables
 
 
 
­

{layers
Etrainable_variables
F	variables
|layer_regularization_losses
Gregularization_losses
}layer_metrics
~metrics
non_trainable_variables
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
F
0
1
2
3
4
5
6
7
	8

9
 
 

0
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
8

total

count
	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
}{
VARIABLE_VALUEAdam/noisy_dense/w_mu/mPlayer_with_weights-0/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/noisy_dense/w_sigma/mSlayer_with_weights-0/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/noisy_dense/b_mu/mPlayer_with_weights-0/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/noisy_dense/b_sigma/mSlayer_with_weights-0/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/noisy_dense_1/w_mu/mPlayer_with_weights-1/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/noisy_dense_1/w_sigma/mSlayer_with_weights-1/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/noisy_dense_1/b_mu/mPlayer_with_weights-1/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/noisy_dense_1/b_sigma/mSlayer_with_weights-1/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/noisy_dense_2/w_mu/mPlayer_with_weights-2/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/noisy_dense_2/w_sigma/mSlayer_with_weights-2/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/noisy_dense_2/b_mu/mPlayer_with_weights-2/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/noisy_dense_2/b_sigma/mSlayer_with_weights-2/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/noisy_dense_3/w_mu/mPlayer_with_weights-3/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/noisy_dense_3/w_sigma/mSlayer_with_weights-3/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/noisy_dense_3/b_mu/mPlayer_with_weights-3/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/noisy_dense_3/b_sigma/mSlayer_with_weights-3/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/categorical_dense/w_mu/mPlayer_with_weights-4/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/categorical_dense/w_sigma/mSlayer_with_weights-4/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/categorical_dense/b_mu/mPlayer_with_weights-4/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/categorical_dense/b_sigma/mSlayer_with_weights-4/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/noisy_dense/w_mu/vPlayer_with_weights-0/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/noisy_dense/w_sigma/vSlayer_with_weights-0/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/noisy_dense/b_mu/vPlayer_with_weights-0/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/noisy_dense/b_sigma/vSlayer_with_weights-0/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/noisy_dense_1/w_mu/vPlayer_with_weights-1/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/noisy_dense_1/w_sigma/vSlayer_with_weights-1/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/noisy_dense_1/b_mu/vPlayer_with_weights-1/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/noisy_dense_1/b_sigma/vSlayer_with_weights-1/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/noisy_dense_2/w_mu/vPlayer_with_weights-2/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/noisy_dense_2/w_sigma/vSlayer_with_weights-2/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/noisy_dense_2/b_mu/vPlayer_with_weights-2/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/noisy_dense_2/b_sigma/vSlayer_with_weights-2/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/noisy_dense_3/w_mu/vPlayer_with_weights-3/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/noisy_dense_3/w_sigma/vSlayer_with_weights-3/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/noisy_dense_3/b_mu/vPlayer_with_weights-3/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/noisy_dense_3/b_sigma/vSlayer_with_weights-3/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/categorical_dense/w_mu/vPlayer_with_weights-4/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/categorical_dense/w_sigma/vSlayer_with_weights-4/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/categorical_dense/b_mu/vPlayer_with_weights-4/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/categorical_dense/b_sigma/vSlayer_with_weights-4/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1noisy_dense/w_munoisy_dense/b_munoisy_dense_1/w_munoisy_dense_1/b_munoisy_dense_2/w_munoisy_dense_2/b_munoisy_dense_3/w_munoisy_dense_3/b_mucategorical_dense/w_mucategorical_dense/b_mu*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_21167886
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$noisy_dense/w_mu/Read/ReadVariableOp'noisy_dense/w_sigma/Read/ReadVariableOp$noisy_dense/b_mu/Read/ReadVariableOp'noisy_dense/b_sigma/Read/ReadVariableOp&noisy_dense_1/w_mu/Read/ReadVariableOp)noisy_dense_1/w_sigma/Read/ReadVariableOp&noisy_dense_1/b_mu/Read/ReadVariableOp)noisy_dense_1/b_sigma/Read/ReadVariableOp&noisy_dense_2/w_mu/Read/ReadVariableOp)noisy_dense_2/w_sigma/Read/ReadVariableOp&noisy_dense_2/b_mu/Read/ReadVariableOp)noisy_dense_2/b_sigma/Read/ReadVariableOp&noisy_dense_3/w_mu/Read/ReadVariableOp)noisy_dense_3/w_sigma/Read/ReadVariableOp&noisy_dense_3/b_mu/Read/ReadVariableOp)noisy_dense_3/b_sigma/Read/ReadVariableOp*categorical_dense/w_mu/Read/ReadVariableOp-categorical_dense/w_sigma/Read/ReadVariableOp*categorical_dense/b_mu/Read/ReadVariableOp-categorical_dense/b_sigma/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/noisy_dense/w_mu/m/Read/ReadVariableOp.Adam/noisy_dense/w_sigma/m/Read/ReadVariableOp+Adam/noisy_dense/b_mu/m/Read/ReadVariableOp.Adam/noisy_dense/b_sigma/m/Read/ReadVariableOp-Adam/noisy_dense_1/w_mu/m/Read/ReadVariableOp0Adam/noisy_dense_1/w_sigma/m/Read/ReadVariableOp-Adam/noisy_dense_1/b_mu/m/Read/ReadVariableOp0Adam/noisy_dense_1/b_sigma/m/Read/ReadVariableOp-Adam/noisy_dense_2/w_mu/m/Read/ReadVariableOp0Adam/noisy_dense_2/w_sigma/m/Read/ReadVariableOp-Adam/noisy_dense_2/b_mu/m/Read/ReadVariableOp0Adam/noisy_dense_2/b_sigma/m/Read/ReadVariableOp-Adam/noisy_dense_3/w_mu/m/Read/ReadVariableOp0Adam/noisy_dense_3/w_sigma/m/Read/ReadVariableOp-Adam/noisy_dense_3/b_mu/m/Read/ReadVariableOp0Adam/noisy_dense_3/b_sigma/m/Read/ReadVariableOp1Adam/categorical_dense/w_mu/m/Read/ReadVariableOp4Adam/categorical_dense/w_sigma/m/Read/ReadVariableOp1Adam/categorical_dense/b_mu/m/Read/ReadVariableOp4Adam/categorical_dense/b_sigma/m/Read/ReadVariableOp+Adam/noisy_dense/w_mu/v/Read/ReadVariableOp.Adam/noisy_dense/w_sigma/v/Read/ReadVariableOp+Adam/noisy_dense/b_mu/v/Read/ReadVariableOp.Adam/noisy_dense/b_sigma/v/Read/ReadVariableOp-Adam/noisy_dense_1/w_mu/v/Read/ReadVariableOp0Adam/noisy_dense_1/w_sigma/v/Read/ReadVariableOp-Adam/noisy_dense_1/b_mu/v/Read/ReadVariableOp0Adam/noisy_dense_1/b_sigma/v/Read/ReadVariableOp-Adam/noisy_dense_2/w_mu/v/Read/ReadVariableOp0Adam/noisy_dense_2/w_sigma/v/Read/ReadVariableOp-Adam/noisy_dense_2/b_mu/v/Read/ReadVariableOp0Adam/noisy_dense_2/b_sigma/v/Read/ReadVariableOp-Adam/noisy_dense_3/w_mu/v/Read/ReadVariableOp0Adam/noisy_dense_3/w_sigma/v/Read/ReadVariableOp-Adam/noisy_dense_3/b_mu/v/Read/ReadVariableOp0Adam/noisy_dense_3/b_sigma/v/Read/ReadVariableOp1Adam/categorical_dense/w_mu/v/Read/ReadVariableOp4Adam/categorical_dense/w_sigma/v/Read/ReadVariableOp1Adam/categorical_dense/b_mu/v/Read/ReadVariableOp4Adam/categorical_dense/b_sigma/v/Read/ReadVariableOpConst_10*P
TinI
G2E	*
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
GPU 2J 8 **
f%R#
!__inference__traced_save_21168745
Ô
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamenoisy_dense/w_munoisy_dense/w_sigmanoisy_dense/b_munoisy_dense/b_sigmanoisy_dense_1/w_munoisy_dense_1/w_sigmanoisy_dense_1/b_munoisy_dense_1/b_sigmanoisy_dense_2/w_munoisy_dense_2/w_sigmanoisy_dense_2/b_munoisy_dense_2/b_sigmanoisy_dense_3/w_munoisy_dense_3/w_sigmanoisy_dense_3/b_munoisy_dense_3/b_sigmacategorical_dense/w_mucategorical_dense/w_sigmacategorical_dense/b_mucategorical_dense/b_sigmabeta_1beta_2decaylearning_rate	Adam/itertotalcountAdam/noisy_dense/w_mu/mAdam/noisy_dense/w_sigma/mAdam/noisy_dense/b_mu/mAdam/noisy_dense/b_sigma/mAdam/noisy_dense_1/w_mu/mAdam/noisy_dense_1/w_sigma/mAdam/noisy_dense_1/b_mu/mAdam/noisy_dense_1/b_sigma/mAdam/noisy_dense_2/w_mu/mAdam/noisy_dense_2/w_sigma/mAdam/noisy_dense_2/b_mu/mAdam/noisy_dense_2/b_sigma/mAdam/noisy_dense_3/w_mu/mAdam/noisy_dense_3/w_sigma/mAdam/noisy_dense_3/b_mu/mAdam/noisy_dense_3/b_sigma/mAdam/categorical_dense/w_mu/m Adam/categorical_dense/w_sigma/mAdam/categorical_dense/b_mu/m Adam/categorical_dense/b_sigma/mAdam/noisy_dense/w_mu/vAdam/noisy_dense/w_sigma/vAdam/noisy_dense/b_mu/vAdam/noisy_dense/b_sigma/vAdam/noisy_dense_1/w_mu/vAdam/noisy_dense_1/w_sigma/vAdam/noisy_dense_1/b_mu/vAdam/noisy_dense_1/b_sigma/vAdam/noisy_dense_2/w_mu/vAdam/noisy_dense_2/w_sigma/vAdam/noisy_dense_2/b_mu/vAdam/noisy_dense_2/b_sigma/vAdam/noisy_dense_3/w_mu/vAdam/noisy_dense_3/w_sigma/vAdam/noisy_dense_3/b_mu/vAdam/noisy_dense_3/b_sigma/vAdam/categorical_dense/w_mu/v Adam/categorical_dense/w_sigma/vAdam/categorical_dense/b_mu/v Adam/categorical_dense/b_sigma/v*O
TinH
F2D*
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
GPU 2J 8 *-
f(R&
$__inference__traced_restore_21168956ÊÐ
ç

.__inference_noisy_dense_layer_call_fn_21168204

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_noisy_dense_layer_call_and_return_conditional_losses_211672002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó

4__inference_categorical_dense_layer_call_fn_21168461

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_categorical_dense_layer_call_and_return_conditional_losses_211675232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
ä
K__inference_noisy_dense_3_layer_call_and_return_conditional_losses_21167415

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 3*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:3*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Í
æ
O__inference_categorical_dense_layer_call_and_return_conditional_losses_21167513

inputs
mul_readvariableop_resource	
mul_y
add_readvariableop_resource!
mul_1_readvariableop_resource
mul_1_y!
add_1_readvariableop_resource
identity¢Add/ReadVariableOp¢Add_1/ReadVariableOp¢Mul/ReadVariableOp¢Mul_1/ReadVariableOp
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

: 3*
dtype02
Mul/ReadVariableOp]
MulMulMul/ReadVariableOp:value:0mul_y*
T0*
_output_shapes

: 32
Mul
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

: 3*
dtype02
Add/ReadVariableOp_
AddAddAdd/ReadVariableOp:value:0Mul:z:0*
T0*
_output_shapes

: 32
Add]
MatMulMatMulinputsAdd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
MatMul
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
:3*
dtype02
Mul_1/ReadVariableOpa
Mul_1MulMul_1/ReadVariableOp:value:0mul_1_y*
T0*
_output_shapes
:32
Mul_1
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:3*
dtype02
Add_1/ReadVariableOpc
Add_1AddAdd_1/ReadVariableOp:value:0	Mul_1:z:0*
T0*
_output_shapes
:32
Add_1l
BiasAddBiasAddMatMul:product:0	Add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32	
BiasAdd¼
IdentityIdentityBiasAdd:output:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Mul/ReadVariableOp^Mul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ :: 3:::3:2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:$ 

_output_shapes

: 3: 

_output_shapes
:3
+
ä
___inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_and_return_conditional_losses_21167828

inputs
noisy_dense_21167798
noisy_dense_21167800
noisy_dense_1_21167803
noisy_dense_1_21167805
noisy_dense_2_21167808
noisy_dense_2_21167810
noisy_dense_3_21167813
noisy_dense_3_21167815
categorical_dense_21167820
categorical_dense_21167822
identity¢)categorical_dense/StatefulPartitionedCall¢#noisy_dense/StatefulPartitionedCall¢%noisy_dense_1/StatefulPartitionedCall¢%noisy_dense_2/StatefulPartitionedCall¢%noisy_dense_3/StatefulPartitionedCall©
#noisy_dense/StatefulPartitionedCallStatefulPartitionedCallinputsnoisy_dense_21167798noisy_dense_21167800*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_noisy_dense_layer_call_and_return_conditional_losses_211672002%
#noisy_dense/StatefulPartitionedCallÙ
%noisy_dense_1/StatefulPartitionedCallStatefulPartitionedCall,noisy_dense/StatefulPartitionedCall:output:0noisy_dense_1_21167803noisy_dense_1_21167805*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_noisy_dense_1_layer_call_and_return_conditional_losses_211672732'
%noisy_dense_1/StatefulPartitionedCallÛ
%noisy_dense_2/StatefulPartitionedCallStatefulPartitionedCall.noisy_dense_1/StatefulPartitionedCall:output:0noisy_dense_2_21167808noisy_dense_2_21167810*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_noisy_dense_2_layer_call_and_return_conditional_losses_211673442'
%noisy_dense_2/StatefulPartitionedCallÛ
%noisy_dense_3/StatefulPartitionedCallStatefulPartitionedCall.noisy_dense_1/StatefulPartitionedCall:output:0noisy_dense_3_21167813noisy_dense_3_21167815*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_noisy_dense_3_layer_call_and_return_conditional_losses_211674152'
%noisy_dense_3/StatefulPartitionedCall¸
concatenate/PartitionedCallPartitionedCall.noisy_dense_2/StatefulPartitionedCall:output:0.noisy_dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_211674632
concatenate/PartitionedCallõ
reshape/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_211674852
reshape/PartitionedCallï
)categorical_dense/StatefulPartitionedCallStatefulPartitionedCall.noisy_dense_1/StatefulPartitionedCall:output:0categorical_dense_21167820categorical_dense_21167822*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_categorical_dense_layer_call_and_return_conditional_losses_211675232+
)categorical_dense/StatefulPartitionedCall£
lambda/PartitionedCallPartitionedCall reshape/PartitionedCall:output:02categorical_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_211675862
lambda/PartitionedCallù
activation/PartitionedCallPartitionedCalllambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_211676122
activation/PartitionedCallÅ
IdentityIdentity#activation/PartitionedCall:output:0*^categorical_dense/StatefulPartitionedCall$^noisy_dense/StatefulPartitionedCall&^noisy_dense_1/StatefulPartitionedCall&^noisy_dense_2/StatefulPartitionedCall&^noisy_dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::::2V
)categorical_dense/StatefulPartitionedCall)categorical_dense/StatefulPartitionedCall2J
#noisy_dense/StatefulPartitionedCall#noisy_dense/StatefulPartitionedCall2N
%noisy_dense_1/StatefulPartitionedCall%noisy_dense_1/StatefulPartitionedCall2N
%noisy_dense_2/StatefulPartitionedCall%noisy_dense_2/StatefulPartitionedCall2N
%noisy_dense_3/StatefulPartitionedCall%noisy_dense_3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë

0__inference_noisy_dense_3_layer_call_fn_21168374

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_noisy_dense_3_layer_call_and_return_conditional_losses_211674152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¥
¦
___inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_and_return_conditional_losses_21167996

inputs+
'noisy_dense_mul_readvariableop_resource
noisy_dense_mul_y+
'noisy_dense_add_readvariableop_resource-
)noisy_dense_mul_1_readvariableop_resource
noisy_dense_mul_1_y-
)noisy_dense_add_1_readvariableop_resource-
)noisy_dense_1_mul_readvariableop_resource
noisy_dense_1_mul_y-
)noisy_dense_1_add_readvariableop_resource/
+noisy_dense_1_mul_1_readvariableop_resource
noisy_dense_1_mul_1_y/
+noisy_dense_1_add_1_readvariableop_resource-
)noisy_dense_2_mul_readvariableop_resource
noisy_dense_2_mul_y-
)noisy_dense_2_add_readvariableop_resource/
+noisy_dense_2_mul_1_readvariableop_resource
noisy_dense_2_mul_1_y/
+noisy_dense_2_add_1_readvariableop_resource-
)noisy_dense_3_mul_readvariableop_resource
noisy_dense_3_mul_y-
)noisy_dense_3_add_readvariableop_resource/
+noisy_dense_3_mul_1_readvariableop_resource
noisy_dense_3_mul_1_y/
+noisy_dense_3_add_1_readvariableop_resource1
-categorical_dense_mul_readvariableop_resource
categorical_dense_mul_y1
-categorical_dense_add_readvariableop_resource3
/categorical_dense_mul_1_readvariableop_resource
categorical_dense_mul_1_y3
/categorical_dense_add_1_readvariableop_resource
identity¢$categorical_dense/Add/ReadVariableOp¢&categorical_dense/Add_1/ReadVariableOp¢$categorical_dense/Mul/ReadVariableOp¢&categorical_dense/Mul_1/ReadVariableOp¢noisy_dense/Add/ReadVariableOp¢ noisy_dense/Add_1/ReadVariableOp¢noisy_dense/Mul/ReadVariableOp¢ noisy_dense/Mul_1/ReadVariableOp¢ noisy_dense_1/Add/ReadVariableOp¢"noisy_dense_1/Add_1/ReadVariableOp¢ noisy_dense_1/Mul/ReadVariableOp¢"noisy_dense_1/Mul_1/ReadVariableOp¢ noisy_dense_2/Add/ReadVariableOp¢"noisy_dense_2/Add_1/ReadVariableOp¢ noisy_dense_2/Mul/ReadVariableOp¢"noisy_dense_2/Mul_1/ReadVariableOp¢ noisy_dense_3/Add/ReadVariableOp¢"noisy_dense_3/Add_1/ReadVariableOp¢ noisy_dense_3/Mul/ReadVariableOp¢"noisy_dense_3/Mul_1/ReadVariableOp¨
noisy_dense/Mul/ReadVariableOpReadVariableOp'noisy_dense_mul_readvariableop_resource*
_output_shapes

:*
dtype02 
noisy_dense/Mul/ReadVariableOp
noisy_dense/MulMul&noisy_dense/Mul/ReadVariableOp:value:0noisy_dense_mul_y*
T0*
_output_shapes

:2
noisy_dense/Mul¨
noisy_dense/Add/ReadVariableOpReadVariableOp'noisy_dense_add_readvariableop_resource*
_output_shapes

:*
dtype02 
noisy_dense/Add/ReadVariableOp
noisy_dense/AddAdd&noisy_dense/Add/ReadVariableOp:value:0noisy_dense/Mul:z:0*
T0*
_output_shapes

:2
noisy_dense/Add
noisy_dense/MatMulMatMulinputsnoisy_dense/Add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
noisy_dense/MatMulª
 noisy_dense/Mul_1/ReadVariableOpReadVariableOp)noisy_dense_mul_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 noisy_dense/Mul_1/ReadVariableOp
noisy_dense/Mul_1Mul(noisy_dense/Mul_1/ReadVariableOp:value:0noisy_dense_mul_1_y*
T0*
_output_shapes
:2
noisy_dense/Mul_1ª
 noisy_dense/Add_1/ReadVariableOpReadVariableOp)noisy_dense_add_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 noisy_dense/Add_1/ReadVariableOp
noisy_dense/Add_1Add(noisy_dense/Add_1/ReadVariableOp:value:0noisy_dense/Mul_1:z:0*
T0*
_output_shapes
:2
noisy_dense/Add_1
noisy_dense/BiasAddBiasAddnoisy_dense/MatMul:product:0noisy_dense/Add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
noisy_dense/BiasAdd|
noisy_dense/ReluRelunoisy_dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
noisy_dense/Relu®
 noisy_dense_1/Mul/ReadVariableOpReadVariableOp)noisy_dense_1_mul_readvariableop_resource*
_output_shapes

: *
dtype02"
 noisy_dense_1/Mul/ReadVariableOp
noisy_dense_1/MulMul(noisy_dense_1/Mul/ReadVariableOp:value:0noisy_dense_1_mul_y*
T0*
_output_shapes

: 2
noisy_dense_1/Mul®
 noisy_dense_1/Add/ReadVariableOpReadVariableOp)noisy_dense_1_add_readvariableop_resource*
_output_shapes

: *
dtype02"
 noisy_dense_1/Add/ReadVariableOp
noisy_dense_1/AddAdd(noisy_dense_1/Add/ReadVariableOp:value:0noisy_dense_1/Mul:z:0*
T0*
_output_shapes

: 2
noisy_dense_1/Add
noisy_dense_1/MatMulMatMulnoisy_dense/Relu:activations:0noisy_dense_1/Add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
noisy_dense_1/MatMul°
"noisy_dense_1/Mul_1/ReadVariableOpReadVariableOp+noisy_dense_1_mul_1_readvariableop_resource*
_output_shapes
: *
dtype02$
"noisy_dense_1/Mul_1/ReadVariableOp
noisy_dense_1/Mul_1Mul*noisy_dense_1/Mul_1/ReadVariableOp:value:0noisy_dense_1_mul_1_y*
T0*
_output_shapes
: 2
noisy_dense_1/Mul_1°
"noisy_dense_1/Add_1/ReadVariableOpReadVariableOp+noisy_dense_1_add_1_readvariableop_resource*
_output_shapes
: *
dtype02$
"noisy_dense_1/Add_1/ReadVariableOp
noisy_dense_1/Add_1Add*noisy_dense_1/Add_1/ReadVariableOp:value:0noisy_dense_1/Mul_1:z:0*
T0*
_output_shapes
: 2
noisy_dense_1/Add_1¤
noisy_dense_1/BiasAddBiasAddnoisy_dense_1/MatMul:product:0noisy_dense_1/Add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
noisy_dense_1/BiasAdd
noisy_dense_1/ReluRelunoisy_dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
noisy_dense_1/Relu®
 noisy_dense_2/Mul/ReadVariableOpReadVariableOp)noisy_dense_2_mul_readvariableop_resource*
_output_shapes

: 3*
dtype02"
 noisy_dense_2/Mul/ReadVariableOp
noisy_dense_2/MulMul(noisy_dense_2/Mul/ReadVariableOp:value:0noisy_dense_2_mul_y*
T0*
_output_shapes

: 32
noisy_dense_2/Mul®
 noisy_dense_2/Add/ReadVariableOpReadVariableOp)noisy_dense_2_add_readvariableop_resource*
_output_shapes

: 3*
dtype02"
 noisy_dense_2/Add/ReadVariableOp
noisy_dense_2/AddAdd(noisy_dense_2/Add/ReadVariableOp:value:0noisy_dense_2/Mul:z:0*
T0*
_output_shapes

: 32
noisy_dense_2/Add¡
noisy_dense_2/MatMulMatMul noisy_dense_1/Relu:activations:0noisy_dense_2/Add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
noisy_dense_2/MatMul°
"noisy_dense_2/Mul_1/ReadVariableOpReadVariableOp+noisy_dense_2_mul_1_readvariableop_resource*
_output_shapes
:3*
dtype02$
"noisy_dense_2/Mul_1/ReadVariableOp
noisy_dense_2/Mul_1Mul*noisy_dense_2/Mul_1/ReadVariableOp:value:0noisy_dense_2_mul_1_y*
T0*
_output_shapes
:32
noisy_dense_2/Mul_1°
"noisy_dense_2/Add_1/ReadVariableOpReadVariableOp+noisy_dense_2_add_1_readvariableop_resource*
_output_shapes
:3*
dtype02$
"noisy_dense_2/Add_1/ReadVariableOp
noisy_dense_2/Add_1Add*noisy_dense_2/Add_1/ReadVariableOp:value:0noisy_dense_2/Mul_1:z:0*
T0*
_output_shapes
:32
noisy_dense_2/Add_1¤
noisy_dense_2/BiasAddBiasAddnoisy_dense_2/MatMul:product:0noisy_dense_2/Add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
noisy_dense_2/BiasAdd®
 noisy_dense_3/Mul/ReadVariableOpReadVariableOp)noisy_dense_3_mul_readvariableop_resource*
_output_shapes

: 3*
dtype02"
 noisy_dense_3/Mul/ReadVariableOp
noisy_dense_3/MulMul(noisy_dense_3/Mul/ReadVariableOp:value:0noisy_dense_3_mul_y*
T0*
_output_shapes

: 32
noisy_dense_3/Mul®
 noisy_dense_3/Add/ReadVariableOpReadVariableOp)noisy_dense_3_add_readvariableop_resource*
_output_shapes

: 3*
dtype02"
 noisy_dense_3/Add/ReadVariableOp
noisy_dense_3/AddAdd(noisy_dense_3/Add/ReadVariableOp:value:0noisy_dense_3/Mul:z:0*
T0*
_output_shapes

: 32
noisy_dense_3/Add¡
noisy_dense_3/MatMulMatMul noisy_dense_1/Relu:activations:0noisy_dense_3/Add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
noisy_dense_3/MatMul°
"noisy_dense_3/Mul_1/ReadVariableOpReadVariableOp+noisy_dense_3_mul_1_readvariableop_resource*
_output_shapes
:3*
dtype02$
"noisy_dense_3/Mul_1/ReadVariableOp
noisy_dense_3/Mul_1Mul*noisy_dense_3/Mul_1/ReadVariableOp:value:0noisy_dense_3_mul_1_y*
T0*
_output_shapes
:32
noisy_dense_3/Mul_1°
"noisy_dense_3/Add_1/ReadVariableOpReadVariableOp+noisy_dense_3_add_1_readvariableop_resource*
_output_shapes
:3*
dtype02$
"noisy_dense_3/Add_1/ReadVariableOp
noisy_dense_3/Add_1Add*noisy_dense_3/Add_1/ReadVariableOp:value:0noisy_dense_3/Mul_1:z:0*
T0*
_output_shapes
:32
noisy_dense_3/Add_1¤
noisy_dense_3/BiasAddBiasAddnoisy_dense_3/MatMul:product:0noisy_dense_3/Add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
noisy_dense_3/BiasAddt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisÑ
concatenate/concatConcatV2noisy_dense_2/BiasAdd:output:0noisy_dense_3/BiasAdd:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2
concatenate/concati
reshape/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
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
reshape/Reshape/shape/2È
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape 
reshape/ReshapeReshapeconcatenate/concat:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
reshape/Reshapeº
$categorical_dense/Mul/ReadVariableOpReadVariableOp-categorical_dense_mul_readvariableop_resource*
_output_shapes

: 3*
dtype02&
$categorical_dense/Mul/ReadVariableOp¥
categorical_dense/MulMul,categorical_dense/Mul/ReadVariableOp:value:0categorical_dense_mul_y*
T0*
_output_shapes

: 32
categorical_dense/Mulº
$categorical_dense/Add/ReadVariableOpReadVariableOp-categorical_dense_add_readvariableop_resource*
_output_shapes

: 3*
dtype02&
$categorical_dense/Add/ReadVariableOp§
categorical_dense/AddAdd,categorical_dense/Add/ReadVariableOp:value:0categorical_dense/Mul:z:0*
T0*
_output_shapes

: 32
categorical_dense/Add­
categorical_dense/MatMulMatMul noisy_dense_1/Relu:activations:0categorical_dense/Add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
categorical_dense/MatMul¼
&categorical_dense/Mul_1/ReadVariableOpReadVariableOp/categorical_dense_mul_1_readvariableop_resource*
_output_shapes
:3*
dtype02(
&categorical_dense/Mul_1/ReadVariableOp©
categorical_dense/Mul_1Mul.categorical_dense/Mul_1/ReadVariableOp:value:0categorical_dense_mul_1_y*
T0*
_output_shapes
:32
categorical_dense/Mul_1¼
&categorical_dense/Add_1/ReadVariableOpReadVariableOp/categorical_dense_add_1_readvariableop_resource*
_output_shapes
:3*
dtype02(
&categorical_dense/Add_1/ReadVariableOp«
categorical_dense/Add_1Add.categorical_dense/Add_1/ReadVariableOp:value:0categorical_dense/Mul_1:z:0*
T0*
_output_shapes
:32
categorical_dense/Add_1´
categorical_dense/BiasAddBiasAdd"categorical_dense/MatMul:product:0categorical_dense/Add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
categorical_dense/BiasAddp
lambda/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
lambda/ExpandDims/dim®
lambda/ExpandDims
ExpandDims"categorical_dense/BiasAdd:output:0lambda/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
lambda/ExpandDims
lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
lambda/Mean/reduction_indices«
lambda/MeanMeanreshape/Reshape:output:0&lambda/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*
	keep_dims(2
lambda/Mean

lambda/subSubreshape/Reshape:output:0lambda/Mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

lambda/sub

lambda/addAddV2lambda/ExpandDims:output:0lambda/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

lambda/add
 activation/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 activation/Max/reduction_indices©
activation/MaxMaxlambda/add:z:0)activation/Max/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
activation/Max
activation/subSublambda/add:z:0activation/Max:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
activation/subq
activation/ExpExpactivation/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
activation/Exp
 activation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 activation/Sum/reduction_indices­
activation/SumSumactivation/Exp:y:0)activation/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
activation/Sum
activation/truedivRealDivactivation/Exp:y:0activation/Sum:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
activation/truedivÆ
IdentityIdentityactivation/truediv:z:0%^categorical_dense/Add/ReadVariableOp'^categorical_dense/Add_1/ReadVariableOp%^categorical_dense/Mul/ReadVariableOp'^categorical_dense/Mul_1/ReadVariableOp^noisy_dense/Add/ReadVariableOp!^noisy_dense/Add_1/ReadVariableOp^noisy_dense/Mul/ReadVariableOp!^noisy_dense/Mul_1/ReadVariableOp!^noisy_dense_1/Add/ReadVariableOp#^noisy_dense_1/Add_1/ReadVariableOp!^noisy_dense_1/Mul/ReadVariableOp#^noisy_dense_1/Mul_1/ReadVariableOp!^noisy_dense_2/Add/ReadVariableOp#^noisy_dense_2/Add_1/ReadVariableOp!^noisy_dense_2/Mul/ReadVariableOp#^noisy_dense_2/Mul_1/ReadVariableOp!^noisy_dense_3/Add/ReadVariableOp#^noisy_dense_3/Add_1/ReadVariableOp!^noisy_dense_3/Mul/ReadVariableOp#^noisy_dense_3/Mul_1/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*È
_input_shapes¶
³:ÿÿÿÿÿÿÿÿÿ:::::::: ::: ::: 3:::3::: 3:::3::: 3:::3:2L
$categorical_dense/Add/ReadVariableOp$categorical_dense/Add/ReadVariableOp2P
&categorical_dense/Add_1/ReadVariableOp&categorical_dense/Add_1/ReadVariableOp2L
$categorical_dense/Mul/ReadVariableOp$categorical_dense/Mul/ReadVariableOp2P
&categorical_dense/Mul_1/ReadVariableOp&categorical_dense/Mul_1/ReadVariableOp2@
noisy_dense/Add/ReadVariableOpnoisy_dense/Add/ReadVariableOp2D
 noisy_dense/Add_1/ReadVariableOp noisy_dense/Add_1/ReadVariableOp2@
noisy_dense/Mul/ReadVariableOpnoisy_dense/Mul/ReadVariableOp2D
 noisy_dense/Mul_1/ReadVariableOp noisy_dense/Mul_1/ReadVariableOp2D
 noisy_dense_1/Add/ReadVariableOp noisy_dense_1/Add/ReadVariableOp2H
"noisy_dense_1/Add_1/ReadVariableOp"noisy_dense_1/Add_1/ReadVariableOp2D
 noisy_dense_1/Mul/ReadVariableOp noisy_dense_1/Mul/ReadVariableOp2H
"noisy_dense_1/Mul_1/ReadVariableOp"noisy_dense_1/Mul_1/ReadVariableOp2D
 noisy_dense_2/Add/ReadVariableOp noisy_dense_2/Add/ReadVariableOp2H
"noisy_dense_2/Add_1/ReadVariableOp"noisy_dense_2/Add_1/ReadVariableOp2D
 noisy_dense_2/Mul/ReadVariableOp noisy_dense_2/Mul/ReadVariableOp2H
"noisy_dense_2/Mul_1/ReadVariableOp"noisy_dense_2/Mul_1/ReadVariableOp2D
 noisy_dense_3/Add/ReadVariableOp noisy_dense_3/Add/ReadVariableOp2H
"noisy_dense_3/Add_1/ReadVariableOp"noisy_dense_3/Add_1/ReadVariableOp2D
 noisy_dense_3/Mul/ReadVariableOp noisy_dense_3/Mul/ReadVariableOp2H
"noisy_dense_3/Mul_1/ReadVariableOp"noisy_dense_3/Mul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: 3: 

_output_shapes
:3:$ 

_output_shapes

: 3: 

_output_shapes
:3:$ 

_output_shapes

: 3: 

_output_shapes
:3
	
ä
K__inference_noisy_dense_2_layer_call_and_return_conditional_losses_21167344

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 3*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:3*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¥
â
K__inference_noisy_dense_1_layer_call_and_return_conditional_losses_21167262

inputs
mul_readvariableop_resource	
mul_y
add_readvariableop_resource!
mul_1_readvariableop_resource
mul_1_y!
add_1_readvariableop_resource
identity¢Add/ReadVariableOp¢Add_1/ReadVariableOp¢Mul/ReadVariableOp¢Mul_1/ReadVariableOp
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

: *
dtype02
Mul/ReadVariableOp]
MulMulMul/ReadVariableOp:value:0mul_y*
T0*
_output_shapes

: 2
Mul
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

: *
dtype02
Add/ReadVariableOp_
AddAddAdd/ReadVariableOp:value:0Mul:z:0*
T0*
_output_shapes

: 2
Add]
MatMulMatMulinputsAdd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
: *
dtype02
Mul_1/ReadVariableOpa
Mul_1MulMul_1/ReadVariableOp:value:0mul_1_y*
T0*
_output_shapes
: 2
Mul_1
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
: *
dtype02
Add_1/ReadVariableOpc
Add_1AddAdd_1/ReadVariableOp:value:0	Mul_1:z:0*
T0*
_output_shapes
: 2
Add_1l
BiasAddBiasAddMatMul:product:0	Add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu¾
IdentityIdentityRelu:activations:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Mul/ReadVariableOp^Mul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ:: ::: :2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

: : 

_output_shapes
: 
þ
n
D__inference_lambda_layer_call_and_return_conditional_losses_21167586

inputs
inputs_1
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputs_1ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

ExpandDimsr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indices
MeanMeaninputsMean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*
	keep_dims(2
Mean^
subSubinputsMean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
subg
addAddV2ExpandDims:output:0sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ3:ÿÿÿÿÿÿÿÿÿ3:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
 
_user_specified_nameinputs
É
â
K__inference_noisy_dense_2_layer_call_and_return_conditional_losses_21168282

inputs
mul_readvariableop_resource	
mul_y
add_readvariableop_resource!
mul_1_readvariableop_resource
mul_1_y!
add_1_readvariableop_resource
identity¢Add/ReadVariableOp¢Add_1/ReadVariableOp¢Mul/ReadVariableOp¢Mul_1/ReadVariableOp
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

: 3*
dtype02
Mul/ReadVariableOp]
MulMulMul/ReadVariableOp:value:0mul_y*
T0*
_output_shapes

: 32
Mul
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

: 3*
dtype02
Add/ReadVariableOp_
AddAddAdd/ReadVariableOp:value:0Mul:z:0*
T0*
_output_shapes

: 32
Add]
MatMulMatMulinputsAdd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
MatMul
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
:3*
dtype02
Mul_1/ReadVariableOpa
Mul_1MulMul_1/ReadVariableOp:value:0mul_1_y*
T0*
_output_shapes
:32
Mul_1
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:3*
dtype02
Add_1/ReadVariableOpc
Add_1AddAdd_1/ReadVariableOp:value:0	Mul_1:z:0*
T0*
_output_shapes
:32
Add_1l
BiasAddBiasAddMatMul:product:0	Add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32	
BiasAdd¼
IdentityIdentityBiasAdd:output:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Mul/ReadVariableOp^Mul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ :: 3:::3:2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:$ 

_output_shapes

: 3: 

_output_shapes
:3
Ä
ô
&__inference_signature_wrapper_21167886
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_211671642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
	
ä
K__inference_noisy_dense_2_layer_call_and_return_conditional_losses_21168292

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 3*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:3*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ë

0__inference_noisy_dense_1_layer_call_fn_21168262

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_noisy_dense_1_layer_call_and_return_conditional_losses_211672732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²7
	
___inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_and_return_conditional_losses_21167621
input_1
noisy_dense_21167228
noisy_dense_21167230
noisy_dense_21167232
noisy_dense_21167234
noisy_dense_21167236
noisy_dense_21167238
noisy_dense_1_21167301
noisy_dense_1_21167303
noisy_dense_1_21167305
noisy_dense_1_21167307
noisy_dense_1_21167309
noisy_dense_1_21167311
noisy_dense_2_21167372
noisy_dense_2_21167374
noisy_dense_2_21167376
noisy_dense_2_21167378
noisy_dense_2_21167380
noisy_dense_2_21167382
noisy_dense_3_21167443
noisy_dense_3_21167445
noisy_dense_3_21167447
noisy_dense_3_21167449
noisy_dense_3_21167451
noisy_dense_3_21167453
categorical_dense_21167551
categorical_dense_21167553
categorical_dense_21167555
categorical_dense_21167557
categorical_dense_21167559
categorical_dense_21167561
identity¢)categorical_dense/StatefulPartitionedCall¢#noisy_dense/StatefulPartitionedCall¢%noisy_dense_1/StatefulPartitionedCall¢%noisy_dense_2/StatefulPartitionedCall¢%noisy_dense_3/StatefulPartitionedCall
#noisy_dense/StatefulPartitionedCallStatefulPartitionedCallinput_1noisy_dense_21167228noisy_dense_21167230noisy_dense_21167232noisy_dense_21167234noisy_dense_21167236noisy_dense_21167238*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_noisy_dense_layer_call_and_return_conditional_losses_211671892%
#noisy_dense/StatefulPartitionedCall¿
%noisy_dense_1/StatefulPartitionedCallStatefulPartitionedCall,noisy_dense/StatefulPartitionedCall:output:0noisy_dense_1_21167301noisy_dense_1_21167303noisy_dense_1_21167305noisy_dense_1_21167307noisy_dense_1_21167309noisy_dense_1_21167311*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_noisy_dense_1_layer_call_and_return_conditional_losses_211672622'
%noisy_dense_1/StatefulPartitionedCallÁ
%noisy_dense_2/StatefulPartitionedCallStatefulPartitionedCall.noisy_dense_1/StatefulPartitionedCall:output:0noisy_dense_2_21167372noisy_dense_2_21167374noisy_dense_2_21167376noisy_dense_2_21167378noisy_dense_2_21167380noisy_dense_2_21167382*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_noisy_dense_2_layer_call_and_return_conditional_losses_211673342'
%noisy_dense_2/StatefulPartitionedCallÁ
%noisy_dense_3/StatefulPartitionedCallStatefulPartitionedCall.noisy_dense_1/StatefulPartitionedCall:output:0noisy_dense_3_21167443noisy_dense_3_21167445noisy_dense_3_21167447noisy_dense_3_21167449noisy_dense_3_21167451noisy_dense_3_21167453*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_noisy_dense_3_layer_call_and_return_conditional_losses_211674052'
%noisy_dense_3/StatefulPartitionedCall¸
concatenate/PartitionedCallPartitionedCall.noisy_dense_2/StatefulPartitionedCall:output:0.noisy_dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_211674632
concatenate/PartitionedCallõ
reshape/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_211674852
reshape/PartitionedCallå
)categorical_dense/StatefulPartitionedCallStatefulPartitionedCall.noisy_dense_1/StatefulPartitionedCall:output:0categorical_dense_21167551categorical_dense_21167553categorical_dense_21167555categorical_dense_21167557categorical_dense_21167559categorical_dense_21167561*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_categorical_dense_layer_call_and_return_conditional_losses_211675132+
)categorical_dense/StatefulPartitionedCall£
lambda/PartitionedCallPartitionedCall reshape/PartitionedCall:output:02categorical_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_211675752
lambda/PartitionedCallù
activation/PartitionedCallPartitionedCalllambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_211676122
activation/PartitionedCallÅ
IdentityIdentity#activation/PartitionedCall:output:0*^categorical_dense/StatefulPartitionedCall$^noisy_dense/StatefulPartitionedCall&^noisy_dense_1/StatefulPartitionedCall&^noisy_dense_2/StatefulPartitionedCall&^noisy_dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*È
_input_shapes¶
³:ÿÿÿÿÿÿÿÿÿ:::::::: ::: ::: 3:::3::: 3:::3::: 3:::3:2V
)categorical_dense/StatefulPartitionedCall)categorical_dense/StatefulPartitionedCall2J
#noisy_dense/StatefulPartitionedCall#noisy_dense/StatefulPartitionedCall2N
%noisy_dense_1/StatefulPartitionedCall%noisy_dense_1/StatefulPartitionedCall2N
%noisy_dense_2/StatefulPartitionedCall%noisy_dense_2/StatefulPartitionedCall2N
%noisy_dense_3/StatefulPartitionedCall%noisy_dense_3/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: 3: 

_output_shapes
:3:$ 

_output_shapes

: 3: 

_output_shapes
:3:$ 

_output_shapes

: 3: 

_output_shapes
:3
¯7
	
___inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_and_return_conditional_losses_21167730

inputs
noisy_dense_21167660
noisy_dense_21167662
noisy_dense_21167664
noisy_dense_21167666
noisy_dense_21167668
noisy_dense_21167670
noisy_dense_1_21167673
noisy_dense_1_21167675
noisy_dense_1_21167677
noisy_dense_1_21167679
noisy_dense_1_21167681
noisy_dense_1_21167683
noisy_dense_2_21167686
noisy_dense_2_21167688
noisy_dense_2_21167690
noisy_dense_2_21167692
noisy_dense_2_21167694
noisy_dense_2_21167696
noisy_dense_3_21167699
noisy_dense_3_21167701
noisy_dense_3_21167703
noisy_dense_3_21167705
noisy_dense_3_21167707
noisy_dense_3_21167709
categorical_dense_21167714
categorical_dense_21167716
categorical_dense_21167718
categorical_dense_21167720
categorical_dense_21167722
categorical_dense_21167724
identity¢)categorical_dense/StatefulPartitionedCall¢#noisy_dense/StatefulPartitionedCall¢%noisy_dense_1/StatefulPartitionedCall¢%noisy_dense_2/StatefulPartitionedCall¢%noisy_dense_3/StatefulPartitionedCall
#noisy_dense/StatefulPartitionedCallStatefulPartitionedCallinputsnoisy_dense_21167660noisy_dense_21167662noisy_dense_21167664noisy_dense_21167666noisy_dense_21167668noisy_dense_21167670*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_noisy_dense_layer_call_and_return_conditional_losses_211671892%
#noisy_dense/StatefulPartitionedCall¿
%noisy_dense_1/StatefulPartitionedCallStatefulPartitionedCall,noisy_dense/StatefulPartitionedCall:output:0noisy_dense_1_21167673noisy_dense_1_21167675noisy_dense_1_21167677noisy_dense_1_21167679noisy_dense_1_21167681noisy_dense_1_21167683*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_noisy_dense_1_layer_call_and_return_conditional_losses_211672622'
%noisy_dense_1/StatefulPartitionedCallÁ
%noisy_dense_2/StatefulPartitionedCallStatefulPartitionedCall.noisy_dense_1/StatefulPartitionedCall:output:0noisy_dense_2_21167686noisy_dense_2_21167688noisy_dense_2_21167690noisy_dense_2_21167692noisy_dense_2_21167694noisy_dense_2_21167696*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_noisy_dense_2_layer_call_and_return_conditional_losses_211673342'
%noisy_dense_2/StatefulPartitionedCallÁ
%noisy_dense_3/StatefulPartitionedCallStatefulPartitionedCall.noisy_dense_1/StatefulPartitionedCall:output:0noisy_dense_3_21167699noisy_dense_3_21167701noisy_dense_3_21167703noisy_dense_3_21167705noisy_dense_3_21167707noisy_dense_3_21167709*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_noisy_dense_3_layer_call_and_return_conditional_losses_211674052'
%noisy_dense_3/StatefulPartitionedCall¸
concatenate/PartitionedCallPartitionedCall.noisy_dense_2/StatefulPartitionedCall:output:0.noisy_dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_211674632
concatenate/PartitionedCallõ
reshape/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_211674852
reshape/PartitionedCallå
)categorical_dense/StatefulPartitionedCallStatefulPartitionedCall.noisy_dense_1/StatefulPartitionedCall:output:0categorical_dense_21167714categorical_dense_21167716categorical_dense_21167718categorical_dense_21167720categorical_dense_21167722categorical_dense_21167724*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_categorical_dense_layer_call_and_return_conditional_losses_211675132+
)categorical_dense/StatefulPartitionedCall£
lambda/PartitionedCallPartitionedCall reshape/PartitionedCall:output:02categorical_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_211675752
lambda/PartitionedCallù
activation/PartitionedCallPartitionedCalllambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_211676122
activation/PartitionedCallÅ
IdentityIdentity#activation/PartitionedCall:output:0*^categorical_dense/StatefulPartitionedCall$^noisy_dense/StatefulPartitionedCall&^noisy_dense_1/StatefulPartitionedCall&^noisy_dense_2/StatefulPartitionedCall&^noisy_dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*È
_input_shapes¶
³:ÿÿÿÿÿÿÿÿÿ:::::::: ::: ::: 3:::3::: 3:::3::: 3:::3:2V
)categorical_dense/StatefulPartitionedCall)categorical_dense/StatefulPartitionedCall2J
#noisy_dense/StatefulPartitionedCall#noisy_dense/StatefulPartitionedCall2N
%noisy_dense_1/StatefulPartitionedCall%noisy_dense_1/StatefulPartitionedCall2N
%noisy_dense_2/StatefulPartitionedCall%noisy_dense_2/StatefulPartitionedCall2N
%noisy_dense_3/StatefulPartitionedCall%noisy_dense_3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: 3: 

_output_shapes
:3:$ 

_output_shapes

: 3: 

_output_shapes
:3:$ 

_output_shapes

: 3: 

_output_shapes
:3
Þ
a
E__inference_reshape_layer_call_and_return_conditional_losses_21167485

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
strided_slice/stack_2â
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
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿf:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
 
_user_specified_nameinputs
	

D__inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_fn_21168146

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *h
fcRa
___inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_and_return_conditional_losses_211678282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£
à
I__inference_noisy_dense_layer_call_and_return_conditional_losses_21167189

inputs
mul_readvariableop_resource	
mul_y
add_readvariableop_resource!
mul_1_readvariableop_resource
mul_1_y!
add_1_readvariableop_resource
identity¢Add/ReadVariableOp¢Add_1/ReadVariableOp¢Mul/ReadVariableOp¢Mul_1/ReadVariableOp
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

:*
dtype02
Mul/ReadVariableOp]
MulMulMul/ReadVariableOp:value:0mul_y*
T0*
_output_shapes

:2
Mul
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

:*
dtype02
Add/ReadVariableOp_
AddAddAdd/ReadVariableOp:value:0Mul:z:0*
T0*
_output_shapes

:2
Add]
MatMulMatMulinputsAdd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
:*
dtype02
Mul_1/ReadVariableOpa
Mul_1MulMul_1/ReadVariableOp:value:0mul_1_y*
T0*
_output_shapes
:2
Mul_1
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype02
Add_1/ReadVariableOpc
Add_1AddAdd_1/ReadVariableOp:value:0	Mul_1:z:0*
T0*
_output_shapes
:2
Add_1l
BiasAddBiasAddMatMul:product:0	Add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¾
IdentityIdentityRelu:activations:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Mul/ReadVariableOp^Mul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

:: 

_output_shapes
:
ó	
â
I__inference_noisy_dense_layer_call_and_return_conditional_losses_21168178

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡
Z
.__inference_concatenate_layer_call_fn_21168387
inputs_0
inputs_1
identityÔ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_211674632
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ3:ÿÿÿÿÿÿÿÿÿ3:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
"
_user_specified_name
inputs/1
É
â
K__inference_noisy_dense_3_layer_call_and_return_conditional_losses_21167405

inputs
mul_readvariableop_resource	
mul_y
add_readvariableop_resource!
mul_1_readvariableop_resource
mul_1_y!
add_1_readvariableop_resource
identity¢Add/ReadVariableOp¢Add_1/ReadVariableOp¢Mul/ReadVariableOp¢Mul_1/ReadVariableOp
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

: 3*
dtype02
Mul/ReadVariableOp]
MulMulMul/ReadVariableOp:value:0mul_y*
T0*
_output_shapes

: 32
Mul
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

: 3*
dtype02
Add/ReadVariableOp_
AddAddAdd/ReadVariableOp:value:0Mul:z:0*
T0*
_output_shapes

: 32
Add]
MatMulMatMulinputsAdd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
MatMul
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
:3*
dtype02
Mul_1/ReadVariableOpa
Mul_1MulMul_1/ReadVariableOp:value:0mul_1_y*
T0*
_output_shapes
:32
Mul_1
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:3*
dtype02
Add_1/ReadVariableOpc
Add_1AddAdd_1/ReadVariableOp:value:0	Mul_1:z:0*
T0*
_output_shapes
:32
Add_1l
BiasAddBiasAddMatMul:product:0	Add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32	
BiasAdd¼
IdentityIdentityBiasAdd:output:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Mul/ReadVariableOp^Mul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ :: 3:::3:2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:$ 

_output_shapes

: 3: 

_output_shapes
:3
µ
s
I__inference_concatenate_layer_call_and_return_conditional_losses_21167463

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
:ÿÿÿÿÿÿÿÿÿf2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ3:ÿÿÿÿÿÿÿÿÿ3:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
 
_user_specified_nameinputs
Í
Ñ
D__inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_fn_21167793
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *h
fcRa
___inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_and_return_conditional_losses_211677302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*È
_input_shapes¶
³:ÿÿÿÿÿÿÿÿÿ:::::::: ::: ::: 3:::3::: 3:::3::: 3:::3:22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: 3: 

_output_shapes
:3:$ 

_output_shapes

: 3: 

_output_shapes
:3:$ 

_output_shapes

: 3: 

_output_shapes
:3
	
p
D__inference_lambda_layer_call_and_return_conditional_losses_21168472
inputs_0
inputs_1
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputs_1ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

ExpandDimsr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indices
MeanMeaninputs_0Mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*
	keep_dims(2
Mean`
subSubinputs_0Mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
subg
addAddV2ExpandDims:output:0sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ3:ÿÿÿÿÿÿÿÿÿ3:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
"
_user_specified_name
inputs/1
	
ä
K__inference_noisy_dense_3_layer_call_and_return_conditional_losses_21168348

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 3*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:3*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
µ
¶%
$__inference__traced_restore_21168956
file_prefix%
!assignvariableop_noisy_dense_w_mu*
&assignvariableop_1_noisy_dense_w_sigma'
#assignvariableop_2_noisy_dense_b_mu*
&assignvariableop_3_noisy_dense_b_sigma)
%assignvariableop_4_noisy_dense_1_w_mu,
(assignvariableop_5_noisy_dense_1_w_sigma)
%assignvariableop_6_noisy_dense_1_b_mu,
(assignvariableop_7_noisy_dense_1_b_sigma)
%assignvariableop_8_noisy_dense_2_w_mu,
(assignvariableop_9_noisy_dense_2_w_sigma*
&assignvariableop_10_noisy_dense_2_b_mu-
)assignvariableop_11_noisy_dense_2_b_sigma*
&assignvariableop_12_noisy_dense_3_w_mu-
)assignvariableop_13_noisy_dense_3_w_sigma*
&assignvariableop_14_noisy_dense_3_b_mu-
)assignvariableop_15_noisy_dense_3_b_sigma.
*assignvariableop_16_categorical_dense_w_mu1
-assignvariableop_17_categorical_dense_w_sigma.
*assignvariableop_18_categorical_dense_b_mu1
-assignvariableop_19_categorical_dense_b_sigma
assignvariableop_20_beta_1
assignvariableop_21_beta_2
assignvariableop_22_decay%
!assignvariableop_23_learning_rate!
assignvariableop_24_adam_iter
assignvariableop_25_total
assignvariableop_26_count/
+assignvariableop_27_adam_noisy_dense_w_mu_m2
.assignvariableop_28_adam_noisy_dense_w_sigma_m/
+assignvariableop_29_adam_noisy_dense_b_mu_m2
.assignvariableop_30_adam_noisy_dense_b_sigma_m1
-assignvariableop_31_adam_noisy_dense_1_w_mu_m4
0assignvariableop_32_adam_noisy_dense_1_w_sigma_m1
-assignvariableop_33_adam_noisy_dense_1_b_mu_m4
0assignvariableop_34_adam_noisy_dense_1_b_sigma_m1
-assignvariableop_35_adam_noisy_dense_2_w_mu_m4
0assignvariableop_36_adam_noisy_dense_2_w_sigma_m1
-assignvariableop_37_adam_noisy_dense_2_b_mu_m4
0assignvariableop_38_adam_noisy_dense_2_b_sigma_m1
-assignvariableop_39_adam_noisy_dense_3_w_mu_m4
0assignvariableop_40_adam_noisy_dense_3_w_sigma_m1
-assignvariableop_41_adam_noisy_dense_3_b_mu_m4
0assignvariableop_42_adam_noisy_dense_3_b_sigma_m5
1assignvariableop_43_adam_categorical_dense_w_mu_m8
4assignvariableop_44_adam_categorical_dense_w_sigma_m5
1assignvariableop_45_adam_categorical_dense_b_mu_m8
4assignvariableop_46_adam_categorical_dense_b_sigma_m/
+assignvariableop_47_adam_noisy_dense_w_mu_v2
.assignvariableop_48_adam_noisy_dense_w_sigma_v/
+assignvariableop_49_adam_noisy_dense_b_mu_v2
.assignvariableop_50_adam_noisy_dense_b_sigma_v1
-assignvariableop_51_adam_noisy_dense_1_w_mu_v4
0assignvariableop_52_adam_noisy_dense_1_w_sigma_v1
-assignvariableop_53_adam_noisy_dense_1_b_mu_v4
0assignvariableop_54_adam_noisy_dense_1_b_sigma_v1
-assignvariableop_55_adam_noisy_dense_2_w_mu_v4
0assignvariableop_56_adam_noisy_dense_2_w_sigma_v1
-assignvariableop_57_adam_noisy_dense_2_b_mu_v4
0assignvariableop_58_adam_noisy_dense_2_b_sigma_v1
-assignvariableop_59_adam_noisy_dense_3_w_mu_v4
0assignvariableop_60_adam_noisy_dense_3_w_sigma_v1
-assignvariableop_61_adam_noisy_dense_3_b_mu_v4
0assignvariableop_62_adam_noisy_dense_3_b_sigma_v5
1assignvariableop_63_adam_categorical_dense_w_mu_v8
4assignvariableop_64_adam_categorical_dense_w_sigma_v5
1assignvariableop_65_adam_categorical_dense_b_mu_v8
4assignvariableop_66_adam_categorical_dense_b_sigma_v
identity_68¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ú&
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*æ%
valueÜ%BÙ%DB4layer_with_weights-0/w_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-0/w_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/b_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-0/b_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/w_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-1/w_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/b_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-1/b_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/w_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-2/w_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/b_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-2/b_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/w_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-3/w_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/b_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-3/b_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/w_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-4/w_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/b_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-4/b_sigma/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¦
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_noisy_dense_w_muIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1«
AssignVariableOp_1AssignVariableOp&assignvariableop_1_noisy_dense_w_sigmaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_noisy_dense_b_muIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3«
AssignVariableOp_3AssignVariableOp&assignvariableop_3_noisy_dense_b_sigmaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ª
AssignVariableOp_4AssignVariableOp%assignvariableop_4_noisy_dense_1_w_muIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5­
AssignVariableOp_5AssignVariableOp(assignvariableop_5_noisy_dense_1_w_sigmaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ª
AssignVariableOp_6AssignVariableOp%assignvariableop_6_noisy_dense_1_b_muIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7­
AssignVariableOp_7AssignVariableOp(assignvariableop_7_noisy_dense_1_b_sigmaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ª
AssignVariableOp_8AssignVariableOp%assignvariableop_8_noisy_dense_2_w_muIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9­
AssignVariableOp_9AssignVariableOp(assignvariableop_9_noisy_dense_2_w_sigmaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10®
AssignVariableOp_10AssignVariableOp&assignvariableop_10_noisy_dense_2_b_muIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11±
AssignVariableOp_11AssignVariableOp)assignvariableop_11_noisy_dense_2_b_sigmaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12®
AssignVariableOp_12AssignVariableOp&assignvariableop_12_noisy_dense_3_w_muIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13±
AssignVariableOp_13AssignVariableOp)assignvariableop_13_noisy_dense_3_w_sigmaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14®
AssignVariableOp_14AssignVariableOp&assignvariableop_14_noisy_dense_3_b_muIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15±
AssignVariableOp_15AssignVariableOp)assignvariableop_15_noisy_dense_3_b_sigmaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16²
AssignVariableOp_16AssignVariableOp*assignvariableop_16_categorical_dense_w_muIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17µ
AssignVariableOp_17AssignVariableOp-assignvariableop_17_categorical_dense_w_sigmaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18²
AssignVariableOp_18AssignVariableOp*assignvariableop_18_categorical_dense_b_muIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19µ
AssignVariableOp_19AssignVariableOp-assignvariableop_19_categorical_dense_b_sigmaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¢
AssignVariableOp_20AssignVariableOpassignvariableop_20_beta_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¢
AssignVariableOp_21AssignVariableOpassignvariableop_21_beta_2Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¡
AssignVariableOp_22AssignVariableOpassignvariableop_22_decayIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23©
AssignVariableOp_23AssignVariableOp!assignvariableop_23_learning_rateIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_24¥
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_iterIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¡
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¡
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27³
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_noisy_dense_w_mu_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¶
AssignVariableOp_28AssignVariableOp.assignvariableop_28_adam_noisy_dense_w_sigma_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29³
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_noisy_dense_b_mu_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¶
AssignVariableOp_30AssignVariableOp.assignvariableop_30_adam_noisy_dense_b_sigma_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31µ
AssignVariableOp_31AssignVariableOp-assignvariableop_31_adam_noisy_dense_1_w_mu_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¸
AssignVariableOp_32AssignVariableOp0assignvariableop_32_adam_noisy_dense_1_w_sigma_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33µ
AssignVariableOp_33AssignVariableOp-assignvariableop_33_adam_noisy_dense_1_b_mu_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34¸
AssignVariableOp_34AssignVariableOp0assignvariableop_34_adam_noisy_dense_1_b_sigma_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35µ
AssignVariableOp_35AssignVariableOp-assignvariableop_35_adam_noisy_dense_2_w_mu_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¸
AssignVariableOp_36AssignVariableOp0assignvariableop_36_adam_noisy_dense_2_w_sigma_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37µ
AssignVariableOp_37AssignVariableOp-assignvariableop_37_adam_noisy_dense_2_b_mu_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38¸
AssignVariableOp_38AssignVariableOp0assignvariableop_38_adam_noisy_dense_2_b_sigma_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39µ
AssignVariableOp_39AssignVariableOp-assignvariableop_39_adam_noisy_dense_3_w_mu_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40¸
AssignVariableOp_40AssignVariableOp0assignvariableop_40_adam_noisy_dense_3_w_sigma_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41µ
AssignVariableOp_41AssignVariableOp-assignvariableop_41_adam_noisy_dense_3_b_mu_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42¸
AssignVariableOp_42AssignVariableOp0assignvariableop_42_adam_noisy_dense_3_b_sigma_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43¹
AssignVariableOp_43AssignVariableOp1assignvariableop_43_adam_categorical_dense_w_mu_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¼
AssignVariableOp_44AssignVariableOp4assignvariableop_44_adam_categorical_dense_w_sigma_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45¹
AssignVariableOp_45AssignVariableOp1assignvariableop_45_adam_categorical_dense_b_mu_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46¼
AssignVariableOp_46AssignVariableOp4assignvariableop_46_adam_categorical_dense_b_sigma_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47³
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_noisy_dense_w_mu_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48¶
AssignVariableOp_48AssignVariableOp.assignvariableop_48_adam_noisy_dense_w_sigma_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49³
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_noisy_dense_b_mu_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50¶
AssignVariableOp_50AssignVariableOp.assignvariableop_50_adam_noisy_dense_b_sigma_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51µ
AssignVariableOp_51AssignVariableOp-assignvariableop_51_adam_noisy_dense_1_w_mu_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52¸
AssignVariableOp_52AssignVariableOp0assignvariableop_52_adam_noisy_dense_1_w_sigma_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53µ
AssignVariableOp_53AssignVariableOp-assignvariableop_53_adam_noisy_dense_1_b_mu_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54¸
AssignVariableOp_54AssignVariableOp0assignvariableop_54_adam_noisy_dense_1_b_sigma_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55µ
AssignVariableOp_55AssignVariableOp-assignvariableop_55_adam_noisy_dense_2_w_mu_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56¸
AssignVariableOp_56AssignVariableOp0assignvariableop_56_adam_noisy_dense_2_w_sigma_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57µ
AssignVariableOp_57AssignVariableOp-assignvariableop_57_adam_noisy_dense_2_b_mu_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58¸
AssignVariableOp_58AssignVariableOp0assignvariableop_58_adam_noisy_dense_2_b_sigma_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59µ
AssignVariableOp_59AssignVariableOp-assignvariableop_59_adam_noisy_dense_3_w_mu_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60¸
AssignVariableOp_60AssignVariableOp0assignvariableop_60_adam_noisy_dense_3_w_sigma_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61µ
AssignVariableOp_61AssignVariableOp-assignvariableop_61_adam_noisy_dense_3_b_mu_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62¸
AssignVariableOp_62AssignVariableOp0assignvariableop_62_adam_noisy_dense_3_b_sigma_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63¹
AssignVariableOp_63AssignVariableOp1assignvariableop_63_adam_categorical_dense_w_mu_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64¼
AssignVariableOp_64AssignVariableOp4assignvariableop_64_adam_categorical_dense_w_sigma_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65¹
AssignVariableOp_65AssignVariableOp1assignvariableop_65_adam_categorical_dense_b_mu_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66¼
AssignVariableOp_66AssignVariableOp4assignvariableop_66_adam_categorical_dense_b_sigma_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_669
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp 
Identity_67Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_67
Identity_68IdentityIdentity_67:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_68"#
identity_68Identity_68:output:0*£
_input_shapes
: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
	
d
H__inference_activation_layer_call_and_return_conditional_losses_21167612

inputs
identityy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Max/reduction_indices
MaxMaxinputsMax/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Max]
subSubinputsMax:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
subP
ExpExpsub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Sum/reduction_indices
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Sumj
truedivRealDivExp:y:0Sum:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32	
truedivc
IdentityIdentitytruediv:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
 
_user_specified_nameinputs
Á
Å
4__inference_categorical_dense_layer_call_fn_21168452

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_categorical_dense_layer_call_and_return_conditional_losses_211675132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ :: 3:::3:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:$ 

_output_shapes

: 3: 

_output_shapes
:3
É
â
K__inference_noisy_dense_2_layer_call_and_return_conditional_losses_21167334

inputs
mul_readvariableop_resource	
mul_y
add_readvariableop_resource!
mul_1_readvariableop_resource
mul_1_y!
add_1_readvariableop_resource
identity¢Add/ReadVariableOp¢Add_1/ReadVariableOp¢Mul/ReadVariableOp¢Mul_1/ReadVariableOp
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

: 3*
dtype02
Mul/ReadVariableOp]
MulMulMul/ReadVariableOp:value:0mul_y*
T0*
_output_shapes

: 32
Mul
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

: 3*
dtype02
Add/ReadVariableOp_
AddAddAdd/ReadVariableOp:value:0Mul:z:0*
T0*
_output_shapes

: 32
Add]
MatMulMatMulinputsAdd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
MatMul
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
:3*
dtype02
Mul_1/ReadVariableOpa
Mul_1MulMul_1/ReadVariableOp:value:0mul_1_y*
T0*
_output_shapes
:32
Mul_1
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:3*
dtype02
Add_1/ReadVariableOpc
Add_1AddAdd_1/ReadVariableOp:value:0	Mul_1:z:0*
T0*
_output_shapes
:32
Add_1l
BiasAddBiasAddMatMul:product:0	Add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32	
BiasAdd¼
IdentityIdentityBiasAdd:output:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Mul/ReadVariableOp^Mul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ :: 3:::3:2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:$ 

_output_shapes

: 3: 

_output_shapes
:3
¹
Á
0__inference_noisy_dense_1_layer_call_fn_21168253

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_noisy_dense_1_layer_call_and_return_conditional_losses_211672622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ:: ::: :22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

: : 

_output_shapes
: 
¾
u
I__inference_concatenate_layer_call_and_return_conditional_losses_21168381
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ3:ÿÿÿÿÿÿÿÿÿ3:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
"
_user_specified_name
inputs/1
³

!__inference__traced_save_21168745
file_prefix/
+savev2_noisy_dense_w_mu_read_readvariableop2
.savev2_noisy_dense_w_sigma_read_readvariableop/
+savev2_noisy_dense_b_mu_read_readvariableop2
.savev2_noisy_dense_b_sigma_read_readvariableop1
-savev2_noisy_dense_1_w_mu_read_readvariableop4
0savev2_noisy_dense_1_w_sigma_read_readvariableop1
-savev2_noisy_dense_1_b_mu_read_readvariableop4
0savev2_noisy_dense_1_b_sigma_read_readvariableop1
-savev2_noisy_dense_2_w_mu_read_readvariableop4
0savev2_noisy_dense_2_w_sigma_read_readvariableop1
-savev2_noisy_dense_2_b_mu_read_readvariableop4
0savev2_noisy_dense_2_b_sigma_read_readvariableop1
-savev2_noisy_dense_3_w_mu_read_readvariableop4
0savev2_noisy_dense_3_w_sigma_read_readvariableop1
-savev2_noisy_dense_3_b_mu_read_readvariableop4
0savev2_noisy_dense_3_b_sigma_read_readvariableop5
1savev2_categorical_dense_w_mu_read_readvariableop8
4savev2_categorical_dense_w_sigma_read_readvariableop5
1savev2_categorical_dense_b_mu_read_readvariableop8
4savev2_categorical_dense_b_sigma_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_noisy_dense_w_mu_m_read_readvariableop9
5savev2_adam_noisy_dense_w_sigma_m_read_readvariableop6
2savev2_adam_noisy_dense_b_mu_m_read_readvariableop9
5savev2_adam_noisy_dense_b_sigma_m_read_readvariableop8
4savev2_adam_noisy_dense_1_w_mu_m_read_readvariableop;
7savev2_adam_noisy_dense_1_w_sigma_m_read_readvariableop8
4savev2_adam_noisy_dense_1_b_mu_m_read_readvariableop;
7savev2_adam_noisy_dense_1_b_sigma_m_read_readvariableop8
4savev2_adam_noisy_dense_2_w_mu_m_read_readvariableop;
7savev2_adam_noisy_dense_2_w_sigma_m_read_readvariableop8
4savev2_adam_noisy_dense_2_b_mu_m_read_readvariableop;
7savev2_adam_noisy_dense_2_b_sigma_m_read_readvariableop8
4savev2_adam_noisy_dense_3_w_mu_m_read_readvariableop;
7savev2_adam_noisy_dense_3_w_sigma_m_read_readvariableop8
4savev2_adam_noisy_dense_3_b_mu_m_read_readvariableop;
7savev2_adam_noisy_dense_3_b_sigma_m_read_readvariableop<
8savev2_adam_categorical_dense_w_mu_m_read_readvariableop?
;savev2_adam_categorical_dense_w_sigma_m_read_readvariableop<
8savev2_adam_categorical_dense_b_mu_m_read_readvariableop?
;savev2_adam_categorical_dense_b_sigma_m_read_readvariableop6
2savev2_adam_noisy_dense_w_mu_v_read_readvariableop9
5savev2_adam_noisy_dense_w_sigma_v_read_readvariableop6
2savev2_adam_noisy_dense_b_mu_v_read_readvariableop9
5savev2_adam_noisy_dense_b_sigma_v_read_readvariableop8
4savev2_adam_noisy_dense_1_w_mu_v_read_readvariableop;
7savev2_adam_noisy_dense_1_w_sigma_v_read_readvariableop8
4savev2_adam_noisy_dense_1_b_mu_v_read_readvariableop;
7savev2_adam_noisy_dense_1_b_sigma_v_read_readvariableop8
4savev2_adam_noisy_dense_2_w_mu_v_read_readvariableop;
7savev2_adam_noisy_dense_2_w_sigma_v_read_readvariableop8
4savev2_adam_noisy_dense_2_b_mu_v_read_readvariableop;
7savev2_adam_noisy_dense_2_b_sigma_v_read_readvariableop8
4savev2_adam_noisy_dense_3_w_mu_v_read_readvariableop;
7savev2_adam_noisy_dense_3_w_sigma_v_read_readvariableop8
4savev2_adam_noisy_dense_3_b_mu_v_read_readvariableop;
7savev2_adam_noisy_dense_3_b_sigma_v_read_readvariableop<
8savev2_adam_categorical_dense_w_mu_v_read_readvariableop?
;savev2_adam_categorical_dense_w_sigma_v_read_readvariableop<
8savev2_adam_categorical_dense_b_mu_v_read_readvariableop?
;savev2_adam_categorical_dense_b_sigma_v_read_readvariableop
savev2_const_10

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÔ&
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*æ%
valueÜ%BÙ%DB4layer_with_weights-0/w_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-0/w_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/b_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-0/b_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/w_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-1/w_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/b_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-1/b_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/w_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-2/w_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/b_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-2/b_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/w_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-3/w_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/b_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-3/b_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/w_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-4/w_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/b_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-4/b_sigma/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_noisy_dense_w_mu_read_readvariableop.savev2_noisy_dense_w_sigma_read_readvariableop+savev2_noisy_dense_b_mu_read_readvariableop.savev2_noisy_dense_b_sigma_read_readvariableop-savev2_noisy_dense_1_w_mu_read_readvariableop0savev2_noisy_dense_1_w_sigma_read_readvariableop-savev2_noisy_dense_1_b_mu_read_readvariableop0savev2_noisy_dense_1_b_sigma_read_readvariableop-savev2_noisy_dense_2_w_mu_read_readvariableop0savev2_noisy_dense_2_w_sigma_read_readvariableop-savev2_noisy_dense_2_b_mu_read_readvariableop0savev2_noisy_dense_2_b_sigma_read_readvariableop-savev2_noisy_dense_3_w_mu_read_readvariableop0savev2_noisy_dense_3_w_sigma_read_readvariableop-savev2_noisy_dense_3_b_mu_read_readvariableop0savev2_noisy_dense_3_b_sigma_read_readvariableop1savev2_categorical_dense_w_mu_read_readvariableop4savev2_categorical_dense_w_sigma_read_readvariableop1savev2_categorical_dense_b_mu_read_readvariableop4savev2_categorical_dense_b_sigma_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_noisy_dense_w_mu_m_read_readvariableop5savev2_adam_noisy_dense_w_sigma_m_read_readvariableop2savev2_adam_noisy_dense_b_mu_m_read_readvariableop5savev2_adam_noisy_dense_b_sigma_m_read_readvariableop4savev2_adam_noisy_dense_1_w_mu_m_read_readvariableop7savev2_adam_noisy_dense_1_w_sigma_m_read_readvariableop4savev2_adam_noisy_dense_1_b_mu_m_read_readvariableop7savev2_adam_noisy_dense_1_b_sigma_m_read_readvariableop4savev2_adam_noisy_dense_2_w_mu_m_read_readvariableop7savev2_adam_noisy_dense_2_w_sigma_m_read_readvariableop4savev2_adam_noisy_dense_2_b_mu_m_read_readvariableop7savev2_adam_noisy_dense_2_b_sigma_m_read_readvariableop4savev2_adam_noisy_dense_3_w_mu_m_read_readvariableop7savev2_adam_noisy_dense_3_w_sigma_m_read_readvariableop4savev2_adam_noisy_dense_3_b_mu_m_read_readvariableop7savev2_adam_noisy_dense_3_b_sigma_m_read_readvariableop8savev2_adam_categorical_dense_w_mu_m_read_readvariableop;savev2_adam_categorical_dense_w_sigma_m_read_readvariableop8savev2_adam_categorical_dense_b_mu_m_read_readvariableop;savev2_adam_categorical_dense_b_sigma_m_read_readvariableop2savev2_adam_noisy_dense_w_mu_v_read_readvariableop5savev2_adam_noisy_dense_w_sigma_v_read_readvariableop2savev2_adam_noisy_dense_b_mu_v_read_readvariableop5savev2_adam_noisy_dense_b_sigma_v_read_readvariableop4savev2_adam_noisy_dense_1_w_mu_v_read_readvariableop7savev2_adam_noisy_dense_1_w_sigma_v_read_readvariableop4savev2_adam_noisy_dense_1_b_mu_v_read_readvariableop7savev2_adam_noisy_dense_1_b_sigma_v_read_readvariableop4savev2_adam_noisy_dense_2_w_mu_v_read_readvariableop7savev2_adam_noisy_dense_2_w_sigma_v_read_readvariableop4savev2_adam_noisy_dense_2_b_mu_v_read_readvariableop7savev2_adam_noisy_dense_2_b_sigma_v_read_readvariableop4savev2_adam_noisy_dense_3_w_mu_v_read_readvariableop7savev2_adam_noisy_dense_3_w_sigma_v_read_readvariableop4savev2_adam_noisy_dense_3_b_mu_v_read_readvariableop7savev2_adam_noisy_dense_3_b_sigma_v_read_readvariableop8savev2_adam_categorical_dense_w_mu_v_read_readvariableop;savev2_adam_categorical_dense_w_sigma_v_read_readvariableop8savev2_adam_categorical_dense_b_mu_v_read_readvariableop;savev2_adam_categorical_dense_b_sigma_v_read_readvariableopsavev2_const_10"/device:CPU:0*
_output_shapes
 *R
dtypesH
F2D	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*
_input_shapesõ
ò: ::::: : : : : 3: 3:3:3: 3: 3:3:3: 3: 3:3:3: : : : : : : ::::: : : : : 3: 3:3:3: 3: 3:3:3: 3: 3:3:3::::: : : : : 3: 3:3:3: 3: 3:3:3: 3: 3:3:3: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

: :$ 

_output_shapes

: : 

_output_shapes
: : 

_output_shapes
: :$	 

_output_shapes

: 3:$
 

_output_shapes

: 3: 

_output_shapes
:3: 

_output_shapes
:3:$ 

_output_shapes

: 3:$ 

_output_shapes

: 3: 

_output_shapes
:3: 

_output_shapes
:3:$ 

_output_shapes

: 3:$ 

_output_shapes

: 3: 

_output_shapes
:3: 

_output_shapes
:3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
::$  

_output_shapes

: :$! 

_output_shapes

: : "

_output_shapes
: : #

_output_shapes
: :$$ 

_output_shapes

: 3:$% 

_output_shapes

: 3: &

_output_shapes
:3: '

_output_shapes
:3:$( 

_output_shapes

: 3:$) 

_output_shapes

: 3: *

_output_shapes
:3: +

_output_shapes
:3:$, 

_output_shapes

: 3:$- 

_output_shapes

: 3: .

_output_shapes
:3: /

_output_shapes
:3:$0 

_output_shapes

::$1 

_output_shapes

:: 2

_output_shapes
:: 3

_output_shapes
::$4 

_output_shapes

: :$5 

_output_shapes

: : 6

_output_shapes
: : 7

_output_shapes
: :$8 

_output_shapes

: 3:$9 

_output_shapes

: 3: :

_output_shapes
:3: ;

_output_shapes
:3:$< 

_output_shapes

: 3:$= 

_output_shapes

: 3: >

_output_shapes
:3: ?

_output_shapes
:3:$@ 

_output_shapes

: 3:$A 

_output_shapes

: 3: B

_output_shapes
:3: C

_output_shapes
:3:D

_output_shapes
: 
Þ
a
E__inference_reshape_layer_call_and_return_conditional_losses_21168400

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
strided_slice/stack_2â
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
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿf:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
 
_user_specified_nameinputs
§
U
)__inference_lambda_layer_call_fn_21168489
inputs_0
inputs_1
identityÓ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_211675752
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ3:ÿÿÿÿÿÿÿÿÿ3:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
"
_user_specified_name
inputs/1
ó	
â
I__inference_noisy_dense_layer_call_and_return_conditional_losses_21167200

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
p
D__inference_lambda_layer_call_and_return_conditional_losses_21168483
inputs_0
inputs_1
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputs_1ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

ExpandDimsr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indices
MeanMeaninputs_0Mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*
	keep_dims(2
Mean`
subSubinputs_0Mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
subg
addAddV2ExpandDims:output:0sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ3:ÿÿÿÿÿÿÿÿÿ3:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
"
_user_specified_name
inputs/1
	

D__inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_fn_21167851
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *h
fcRa
___inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_and_return_conditional_losses_211678282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
O

___inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_and_return_conditional_losses_21168056

inputs.
*noisy_dense_matmul_readvariableop_resource/
+noisy_dense_biasadd_readvariableop_resource0
,noisy_dense_1_matmul_readvariableop_resource1
-noisy_dense_1_biasadd_readvariableop_resource0
,noisy_dense_2_matmul_readvariableop_resource1
-noisy_dense_2_biasadd_readvariableop_resource0
,noisy_dense_3_matmul_readvariableop_resource1
-noisy_dense_3_biasadd_readvariableop_resource4
0categorical_dense_matmul_readvariableop_resource5
1categorical_dense_biasadd_readvariableop_resource
identity¢(categorical_dense/BiasAdd/ReadVariableOp¢'categorical_dense/MatMul/ReadVariableOp¢"noisy_dense/BiasAdd/ReadVariableOp¢!noisy_dense/MatMul/ReadVariableOp¢$noisy_dense_1/BiasAdd/ReadVariableOp¢#noisy_dense_1/MatMul/ReadVariableOp¢$noisy_dense_2/BiasAdd/ReadVariableOp¢#noisy_dense_2/MatMul/ReadVariableOp¢$noisy_dense_3/BiasAdd/ReadVariableOp¢#noisy_dense_3/MatMul/ReadVariableOp±
!noisy_dense/MatMul/ReadVariableOpReadVariableOp*noisy_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!noisy_dense/MatMul/ReadVariableOp
noisy_dense/MatMulMatMulinputs)noisy_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
noisy_dense/MatMul°
"noisy_dense/BiasAdd/ReadVariableOpReadVariableOp+noisy_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"noisy_dense/BiasAdd/ReadVariableOp±
noisy_dense/BiasAddBiasAddnoisy_dense/MatMul:product:0*noisy_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
noisy_dense/BiasAdd|
noisy_dense/ReluRelunoisy_dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
noisy_dense/Relu·
#noisy_dense_1/MatMul/ReadVariableOpReadVariableOp,noisy_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#noisy_dense_1/MatMul/ReadVariableOpµ
noisy_dense_1/MatMulMatMulnoisy_dense/Relu:activations:0+noisy_dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
noisy_dense_1/MatMul¶
$noisy_dense_1/BiasAdd/ReadVariableOpReadVariableOp-noisy_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$noisy_dense_1/BiasAdd/ReadVariableOp¹
noisy_dense_1/BiasAddBiasAddnoisy_dense_1/MatMul:product:0,noisy_dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
noisy_dense_1/BiasAdd
noisy_dense_1/ReluRelunoisy_dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
noisy_dense_1/Relu·
#noisy_dense_2/MatMul/ReadVariableOpReadVariableOp,noisy_dense_2_matmul_readvariableop_resource*
_output_shapes

: 3*
dtype02%
#noisy_dense_2/MatMul/ReadVariableOp·
noisy_dense_2/MatMulMatMul noisy_dense_1/Relu:activations:0+noisy_dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
noisy_dense_2/MatMul¶
$noisy_dense_2/BiasAdd/ReadVariableOpReadVariableOp-noisy_dense_2_biasadd_readvariableop_resource*
_output_shapes
:3*
dtype02&
$noisy_dense_2/BiasAdd/ReadVariableOp¹
noisy_dense_2/BiasAddBiasAddnoisy_dense_2/MatMul:product:0,noisy_dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
noisy_dense_2/BiasAdd·
#noisy_dense_3/MatMul/ReadVariableOpReadVariableOp,noisy_dense_3_matmul_readvariableop_resource*
_output_shapes

: 3*
dtype02%
#noisy_dense_3/MatMul/ReadVariableOp·
noisy_dense_3/MatMulMatMul noisy_dense_1/Relu:activations:0+noisy_dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
noisy_dense_3/MatMul¶
$noisy_dense_3/BiasAdd/ReadVariableOpReadVariableOp-noisy_dense_3_biasadd_readvariableop_resource*
_output_shapes
:3*
dtype02&
$noisy_dense_3/BiasAdd/ReadVariableOp¹
noisy_dense_3/BiasAddBiasAddnoisy_dense_3/MatMul:product:0,noisy_dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
noisy_dense_3/BiasAddt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisÑ
concatenate/concatConcatV2noisy_dense_2/BiasAdd:output:0noisy_dense_3/BiasAdd:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2
concatenate/concati
reshape/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
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
reshape/Reshape/shape/2È
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape 
reshape/ReshapeReshapeconcatenate/concat:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
reshape/ReshapeÃ
'categorical_dense/MatMul/ReadVariableOpReadVariableOp0categorical_dense_matmul_readvariableop_resource*
_output_shapes

: 3*
dtype02)
'categorical_dense/MatMul/ReadVariableOpÃ
categorical_dense/MatMulMatMul noisy_dense_1/Relu:activations:0/categorical_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
categorical_dense/MatMulÂ
(categorical_dense/BiasAdd/ReadVariableOpReadVariableOp1categorical_dense_biasadd_readvariableop_resource*
_output_shapes
:3*
dtype02*
(categorical_dense/BiasAdd/ReadVariableOpÉ
categorical_dense/BiasAddBiasAdd"categorical_dense/MatMul:product:00categorical_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
categorical_dense/BiasAddp
lambda/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
lambda/ExpandDims/dim®
lambda/ExpandDims
ExpandDims"categorical_dense/BiasAdd:output:0lambda/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
lambda/ExpandDims
lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
lambda/Mean/reduction_indices«
lambda/MeanMeanreshape/Reshape:output:0&lambda/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*
	keep_dims(2
lambda/Mean

lambda/subSubreshape/Reshape:output:0lambda/Mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

lambda/sub

lambda/addAddV2lambda/ExpandDims:output:0lambda/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

lambda/add
 activation/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 activation/Max/reduction_indices©
activation/MaxMaxlambda/add:z:0)activation/Max/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
activation/Max
activation/subSublambda/add:z:0activation/Max:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
activation/subq
activation/ExpExpactivation/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
activation/Exp
 activation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 activation/Sum/reduction_indices­
activation/SumSumactivation/Exp:y:0)activation/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
activation/Sum
activation/truedivRealDivactivation/Exp:y:0activation/Sum:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
activation/truedivó
IdentityIdentityactivation/truediv:z:0)^categorical_dense/BiasAdd/ReadVariableOp(^categorical_dense/MatMul/ReadVariableOp#^noisy_dense/BiasAdd/ReadVariableOp"^noisy_dense/MatMul/ReadVariableOp%^noisy_dense_1/BiasAdd/ReadVariableOp$^noisy_dense_1/MatMul/ReadVariableOp%^noisy_dense_2/BiasAdd/ReadVariableOp$^noisy_dense_2/MatMul/ReadVariableOp%^noisy_dense_3/BiasAdd/ReadVariableOp$^noisy_dense_3/MatMul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::::2T
(categorical_dense/BiasAdd/ReadVariableOp(categorical_dense/BiasAdd/ReadVariableOp2R
'categorical_dense/MatMul/ReadVariableOp'categorical_dense/MatMul/ReadVariableOp2H
"noisy_dense/BiasAdd/ReadVariableOp"noisy_dense/BiasAdd/ReadVariableOp2F
!noisy_dense/MatMul/ReadVariableOp!noisy_dense/MatMul/ReadVariableOp2L
$noisy_dense_1/BiasAdd/ReadVariableOp$noisy_dense_1/BiasAdd/ReadVariableOp2J
#noisy_dense_1/MatMul/ReadVariableOp#noisy_dense_1/MatMul/ReadVariableOp2L
$noisy_dense_2/BiasAdd/ReadVariableOp$noisy_dense_2/BiasAdd/ReadVariableOp2J
#noisy_dense_2/MatMul/ReadVariableOp#noisy_dense_2/MatMul/ReadVariableOp2L
$noisy_dense_3/BiasAdd/ReadVariableOp$noisy_dense_3/BiasAdd/ReadVariableOp2J
#noisy_dense_3/MatMul/ReadVariableOp#noisy_dense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë

0__inference_noisy_dense_2_layer_call_fn_21168318

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_noisy_dense_2_layer_call_and_return_conditional_losses_211673442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
É
â
K__inference_noisy_dense_3_layer_call_and_return_conditional_losses_21168338

inputs
mul_readvariableop_resource	
mul_y
add_readvariableop_resource!
mul_1_readvariableop_resource
mul_1_y!
add_1_readvariableop_resource
identity¢Add/ReadVariableOp¢Add_1/ReadVariableOp¢Mul/ReadVariableOp¢Mul_1/ReadVariableOp
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

: 3*
dtype02
Mul/ReadVariableOp]
MulMulMul/ReadVariableOp:value:0mul_y*
T0*
_output_shapes

: 32
Mul
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

: 3*
dtype02
Add/ReadVariableOp_
AddAddAdd/ReadVariableOp:value:0Mul:z:0*
T0*
_output_shapes

: 32
Add]
MatMulMatMulinputsAdd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
MatMul
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
:3*
dtype02
Mul_1/ReadVariableOpa
Mul_1MulMul_1/ReadVariableOp:value:0mul_1_y*
T0*
_output_shapes
:32
Mul_1
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:3*
dtype02
Add_1/ReadVariableOpc
Add_1AddAdd_1/ReadVariableOp:value:0	Mul_1:z:0*
T0*
_output_shapes
:32
Add_1l
BiasAddBiasAddMatMul:product:0	Add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32	
BiasAdd¼
IdentityIdentityBiasAdd:output:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Mul/ReadVariableOp^Mul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ :: 3:::3:2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:$ 

_output_shapes

: 3: 

_output_shapes
:3

F
*__inference_reshape_layer_call_fn_21168405

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_211674852
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿf:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
 
_user_specified_nameinputs
õ	
ä
K__inference_noisy_dense_1_layer_call_and_return_conditional_losses_21168236

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥
â
K__inference_noisy_dense_1_layer_call_and_return_conditional_losses_21168225

inputs
mul_readvariableop_resource	
mul_y
add_readvariableop_resource!
mul_1_readvariableop_resource
mul_1_y!
add_1_readvariableop_resource
identity¢Add/ReadVariableOp¢Add_1/ReadVariableOp¢Mul/ReadVariableOp¢Mul_1/ReadVariableOp
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

: *
dtype02
Mul/ReadVariableOp]
MulMulMul/ReadVariableOp:value:0mul_y*
T0*
_output_shapes

: 2
Mul
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

: *
dtype02
Add/ReadVariableOp_
AddAddAdd/ReadVariableOp:value:0Mul:z:0*
T0*
_output_shapes

: 2
Add]
MatMulMatMulinputsAdd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
: *
dtype02
Mul_1/ReadVariableOpa
Mul_1MulMul_1/ReadVariableOp:value:0mul_1_y*
T0*
_output_shapes
: 2
Mul_1
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
: *
dtype02
Add_1/ReadVariableOpc
Add_1AddAdd_1/ReadVariableOp:value:0	Mul_1:z:0*
T0*
_output_shapes
: 2
Add_1l
BiasAddBiasAddMatMul:product:0	Add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu¾
IdentityIdentityRelu:activations:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Mul/ReadVariableOp^Mul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ:: ::: :2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

: : 

_output_shapes
: 
	
è
O__inference_categorical_dense_layer_call_and_return_conditional_losses_21167523

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 3*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:3*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
£
à
I__inference_noisy_dense_layer_call_and_return_conditional_losses_21168167

inputs
mul_readvariableop_resource	
mul_y
add_readvariableop_resource!
mul_1_readvariableop_resource
mul_1_y!
add_1_readvariableop_resource
identity¢Add/ReadVariableOp¢Add_1/ReadVariableOp¢Mul/ReadVariableOp¢Mul_1/ReadVariableOp
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

:*
dtype02
Mul/ReadVariableOp]
MulMulMul/ReadVariableOp:value:0mul_y*
T0*
_output_shapes

:2
Mul
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

:*
dtype02
Add/ReadVariableOp_
AddAddAdd/ReadVariableOp:value:0Mul:z:0*
T0*
_output_shapes

:2
Add]
MatMulMatMulinputsAdd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
:*
dtype02
Mul_1/ReadVariableOpa
Mul_1MulMul_1/ReadVariableOp:value:0mul_1_y*
T0*
_output_shapes
:2
Mul_1
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype02
Add_1/ReadVariableOpc
Add_1AddAdd_1/ReadVariableOp:value:0	Mul_1:z:0*
T0*
_output_shapes
:2
Add_1l
BiasAddBiasAddMatMul:product:0	Add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¾
IdentityIdentityRelu:activations:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Mul/ReadVariableOp^Mul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

:: 

_output_shapes
:
+
å
___inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_and_return_conditional_losses_21167654
input_1
noisy_dense_21167624
noisy_dense_21167626
noisy_dense_1_21167629
noisy_dense_1_21167631
noisy_dense_2_21167634
noisy_dense_2_21167636
noisy_dense_3_21167639
noisy_dense_3_21167641
categorical_dense_21167646
categorical_dense_21167648
identity¢)categorical_dense/StatefulPartitionedCall¢#noisy_dense/StatefulPartitionedCall¢%noisy_dense_1/StatefulPartitionedCall¢%noisy_dense_2/StatefulPartitionedCall¢%noisy_dense_3/StatefulPartitionedCallª
#noisy_dense/StatefulPartitionedCallStatefulPartitionedCallinput_1noisy_dense_21167624noisy_dense_21167626*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_noisy_dense_layer_call_and_return_conditional_losses_211672002%
#noisy_dense/StatefulPartitionedCallÙ
%noisy_dense_1/StatefulPartitionedCallStatefulPartitionedCall,noisy_dense/StatefulPartitionedCall:output:0noisy_dense_1_21167629noisy_dense_1_21167631*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_noisy_dense_1_layer_call_and_return_conditional_losses_211672732'
%noisy_dense_1/StatefulPartitionedCallÛ
%noisy_dense_2/StatefulPartitionedCallStatefulPartitionedCall.noisy_dense_1/StatefulPartitionedCall:output:0noisy_dense_2_21167634noisy_dense_2_21167636*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_noisy_dense_2_layer_call_and_return_conditional_losses_211673442'
%noisy_dense_2/StatefulPartitionedCallÛ
%noisy_dense_3/StatefulPartitionedCallStatefulPartitionedCall.noisy_dense_1/StatefulPartitionedCall:output:0noisy_dense_3_21167639noisy_dense_3_21167641*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_noisy_dense_3_layer_call_and_return_conditional_losses_211674152'
%noisy_dense_3/StatefulPartitionedCall¸
concatenate/PartitionedCallPartitionedCall.noisy_dense_2/StatefulPartitionedCall:output:0.noisy_dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_211674632
concatenate/PartitionedCallõ
reshape/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_211674852
reshape/PartitionedCallï
)categorical_dense/StatefulPartitionedCallStatefulPartitionedCall.noisy_dense_1/StatefulPartitionedCall:output:0categorical_dense_21167646categorical_dense_21167648*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_categorical_dense_layer_call_and_return_conditional_losses_211675232+
)categorical_dense/StatefulPartitionedCall£
lambda/PartitionedCallPartitionedCall reshape/PartitionedCall:output:02categorical_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_211675862
lambda/PartitionedCallù
activation/PartitionedCallPartitionedCalllambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_211676122
activation/PartitionedCallÅ
IdentityIdentity#activation/PartitionedCall:output:0*^categorical_dense/StatefulPartitionedCall$^noisy_dense/StatefulPartitionedCall&^noisy_dense_1/StatefulPartitionedCall&^noisy_dense_2/StatefulPartitionedCall&^noisy_dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::::2V
)categorical_dense/StatefulPartitionedCall)categorical_dense/StatefulPartitionedCall2J
#noisy_dense/StatefulPartitionedCall#noisy_dense/StatefulPartitionedCall2N
%noisy_dense_1/StatefulPartitionedCall%noisy_dense_1/StatefulPartitionedCall2N
%noisy_dense_2/StatefulPartitionedCall%noisy_dense_2/StatefulPartitionedCall2N
%noisy_dense_3/StatefulPartitionedCall%noisy_dense_3/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
þ
n
D__inference_lambda_layer_call_and_return_conditional_losses_21167575

inputs
inputs_1
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputs_1ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

ExpandDimsr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indices
MeanMeaninputsMean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*
	keep_dims(2
Mean^
subSubinputsMean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
subg
addAddV2ExpandDims:output:0sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ3:ÿÿÿÿÿÿÿÿÿ3:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
 
_user_specified_nameinputs
	
è
O__inference_categorical_dense_layer_call_and_return_conditional_losses_21168435

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 3*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:3*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ê
Ð
D__inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_fn_21168121

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *h
fcRa
___inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_and_return_conditional_losses_211677302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*È
_input_shapes¶
³:ÿÿÿÿÿÿÿÿÿ:::::::: ::: ::: 3:::3::: 3:::3::: 3:::3:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: 3: 

_output_shapes
:3:$ 

_output_shapes

: 3: 

_output_shapes
:3:$ 

_output_shapes

: 3: 

_output_shapes
:3
§
U
)__inference_lambda_layer_call_fn_21168495
inputs_0
inputs_1
identityÓ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_211675862
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ3:ÿÿÿÿÿÿÿÿÿ3:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
"
_user_specified_name
inputs/1
Í
æ
O__inference_categorical_dense_layer_call_and_return_conditional_losses_21168425

inputs
mul_readvariableop_resource	
mul_y
add_readvariableop_resource!
mul_1_readvariableop_resource
mul_1_y!
add_1_readvariableop_resource
identity¢Add/ReadVariableOp¢Add_1/ReadVariableOp¢Mul/ReadVariableOp¢Mul_1/ReadVariableOp
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

: 3*
dtype02
Mul/ReadVariableOp]
MulMulMul/ReadVariableOp:value:0mul_y*
T0*
_output_shapes

: 32
Mul
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

: 3*
dtype02
Add/ReadVariableOp_
AddAddAdd/ReadVariableOp:value:0Mul:z:0*
T0*
_output_shapes

: 32
Add]
MatMulMatMulinputsAdd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
MatMul
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
:3*
dtype02
Mul_1/ReadVariableOpa
Mul_1MulMul_1/ReadVariableOp:value:0mul_1_y*
T0*
_output_shapes
:32
Mul_1
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:3*
dtype02
Add_1/ReadVariableOpc
Add_1AddAdd_1/ReadVariableOp:value:0	Mul_1:z:0*
T0*
_output_shapes
:32
Add_1l
BiasAddBiasAddMatMul:product:0	Add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32	
BiasAdd¼
IdentityIdentityBiasAdd:output:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Mul/ReadVariableOp^Mul_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ :: 3:::3:2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:$ 

_output_shapes

: 3: 

_output_shapes
:3
µ
¿
.__inference_noisy_dense_layer_call_fn_21168195

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_noisy_dense_layer_call_and_return_conditional_losses_211671892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

:: 

_output_shapes
:
ù
í
#__inference__wrapped_model_21167164
input_1P
Ldqn_noisy_net_categorical_dueling_noisy_dense_matmul_readvariableop_resourceQ
Mdqn_noisy_net_categorical_dueling_noisy_dense_biasadd_readvariableop_resourceR
Ndqn_noisy_net_categorical_dueling_noisy_dense_1_matmul_readvariableop_resourceS
Odqn_noisy_net_categorical_dueling_noisy_dense_1_biasadd_readvariableop_resourceR
Ndqn_noisy_net_categorical_dueling_noisy_dense_2_matmul_readvariableop_resourceS
Odqn_noisy_net_categorical_dueling_noisy_dense_2_biasadd_readvariableop_resourceR
Ndqn_noisy_net_categorical_dueling_noisy_dense_3_matmul_readvariableop_resourceS
Odqn_noisy_net_categorical_dueling_noisy_dense_3_biasadd_readvariableop_resourceV
Rdqn_noisy_net_categorical_dueling_categorical_dense_matmul_readvariableop_resourceW
Sdqn_noisy_net_categorical_dueling_categorical_dense_biasadd_readvariableop_resource
identity¢JDQN_Noisy_Net_Categorical_Dueling/categorical_dense/BiasAdd/ReadVariableOp¢IDQN_Noisy_Net_Categorical_Dueling/categorical_dense/MatMul/ReadVariableOp¢DDQN_Noisy_Net_Categorical_Dueling/noisy_dense/BiasAdd/ReadVariableOp¢CDQN_Noisy_Net_Categorical_Dueling/noisy_dense/MatMul/ReadVariableOp¢FDQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/BiasAdd/ReadVariableOp¢EDQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/MatMul/ReadVariableOp¢FDQN_Noisy_Net_Categorical_Dueling/noisy_dense_2/BiasAdd/ReadVariableOp¢EDQN_Noisy_Net_Categorical_Dueling/noisy_dense_2/MatMul/ReadVariableOp¢FDQN_Noisy_Net_Categorical_Dueling/noisy_dense_3/BiasAdd/ReadVariableOp¢EDQN_Noisy_Net_Categorical_Dueling/noisy_dense_3/MatMul/ReadVariableOp
CDQN_Noisy_Net_Categorical_Dueling/noisy_dense/MatMul/ReadVariableOpReadVariableOpLdqn_noisy_net_categorical_dueling_noisy_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02E
CDQN_Noisy_Net_Categorical_Dueling/noisy_dense/MatMul/ReadVariableOpþ
4DQN_Noisy_Net_Categorical_Dueling/noisy_dense/MatMulMatMulinput_1KDQN_Noisy_Net_Categorical_Dueling/noisy_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4DQN_Noisy_Net_Categorical_Dueling/noisy_dense/MatMul
DDQN_Noisy_Net_Categorical_Dueling/noisy_dense/BiasAdd/ReadVariableOpReadVariableOpMdqn_noisy_net_categorical_dueling_noisy_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02F
DDQN_Noisy_Net_Categorical_Dueling/noisy_dense/BiasAdd/ReadVariableOp¹
5DQN_Noisy_Net_Categorical_Dueling/noisy_dense/BiasAddBiasAdd>DQN_Noisy_Net_Categorical_Dueling/noisy_dense/MatMul:product:0LDQN_Noisy_Net_Categorical_Dueling/noisy_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5DQN_Noisy_Net_Categorical_Dueling/noisy_dense/BiasAddâ
2DQN_Noisy_Net_Categorical_Dueling/noisy_dense/ReluRelu>DQN_Noisy_Net_Categorical_Dueling/noisy_dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2DQN_Noisy_Net_Categorical_Dueling/noisy_dense/Relu
EDQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/MatMul/ReadVariableOpReadVariableOpNdqn_noisy_net_categorical_dueling_noisy_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02G
EDQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/MatMul/ReadVariableOp½
6DQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/MatMulMatMul@DQN_Noisy_Net_Categorical_Dueling/noisy_dense/Relu:activations:0MDQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 28
6DQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/MatMul
FDQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/BiasAdd/ReadVariableOpReadVariableOpOdqn_noisy_net_categorical_dueling_noisy_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02H
FDQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/BiasAdd/ReadVariableOpÁ
7DQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/BiasAddBiasAdd@DQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/MatMul:product:0NDQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 29
7DQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/BiasAddè
4DQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/ReluRelu@DQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 26
4DQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/Relu
EDQN_Noisy_Net_Categorical_Dueling/noisy_dense_2/MatMul/ReadVariableOpReadVariableOpNdqn_noisy_net_categorical_dueling_noisy_dense_2_matmul_readvariableop_resource*
_output_shapes

: 3*
dtype02G
EDQN_Noisy_Net_Categorical_Dueling/noisy_dense_2/MatMul/ReadVariableOp¿
6DQN_Noisy_Net_Categorical_Dueling/noisy_dense_2/MatMulMatMulBDQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/Relu:activations:0MDQN_Noisy_Net_Categorical_Dueling/noisy_dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ328
6DQN_Noisy_Net_Categorical_Dueling/noisy_dense_2/MatMul
FDQN_Noisy_Net_Categorical_Dueling/noisy_dense_2/BiasAdd/ReadVariableOpReadVariableOpOdqn_noisy_net_categorical_dueling_noisy_dense_2_biasadd_readvariableop_resource*
_output_shapes
:3*
dtype02H
FDQN_Noisy_Net_Categorical_Dueling/noisy_dense_2/BiasAdd/ReadVariableOpÁ
7DQN_Noisy_Net_Categorical_Dueling/noisy_dense_2/BiasAddBiasAdd@DQN_Noisy_Net_Categorical_Dueling/noisy_dense_2/MatMul:product:0NDQN_Noisy_Net_Categorical_Dueling/noisy_dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ329
7DQN_Noisy_Net_Categorical_Dueling/noisy_dense_2/BiasAdd
EDQN_Noisy_Net_Categorical_Dueling/noisy_dense_3/MatMul/ReadVariableOpReadVariableOpNdqn_noisy_net_categorical_dueling_noisy_dense_3_matmul_readvariableop_resource*
_output_shapes

: 3*
dtype02G
EDQN_Noisy_Net_Categorical_Dueling/noisy_dense_3/MatMul/ReadVariableOp¿
6DQN_Noisy_Net_Categorical_Dueling/noisy_dense_3/MatMulMatMulBDQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/Relu:activations:0MDQN_Noisy_Net_Categorical_Dueling/noisy_dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ328
6DQN_Noisy_Net_Categorical_Dueling/noisy_dense_3/MatMul
FDQN_Noisy_Net_Categorical_Dueling/noisy_dense_3/BiasAdd/ReadVariableOpReadVariableOpOdqn_noisy_net_categorical_dueling_noisy_dense_3_biasadd_readvariableop_resource*
_output_shapes
:3*
dtype02H
FDQN_Noisy_Net_Categorical_Dueling/noisy_dense_3/BiasAdd/ReadVariableOpÁ
7DQN_Noisy_Net_Categorical_Dueling/noisy_dense_3/BiasAddBiasAdd@DQN_Noisy_Net_Categorical_Dueling/noisy_dense_3/MatMul:product:0NDQN_Noisy_Net_Categorical_Dueling/noisy_dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ329
7DQN_Noisy_Net_Categorical_Dueling/noisy_dense_3/BiasAdd¸
9DQN_Noisy_Net_Categorical_Dueling/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2;
9DQN_Noisy_Net_Categorical_Dueling/concatenate/concat/axisû
4DQN_Noisy_Net_Categorical_Dueling/concatenate/concatConcatV2@DQN_Noisy_Net_Categorical_Dueling/noisy_dense_2/BiasAdd:output:0@DQN_Noisy_Net_Categorical_Dueling/noisy_dense_3/BiasAdd:output:0BDQN_Noisy_Net_Categorical_Dueling/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf26
4DQN_Noisy_Net_Categorical_Dueling/concatenate/concatÏ
/DQN_Noisy_Net_Categorical_Dueling/reshape/ShapeShape=DQN_Noisy_Net_Categorical_Dueling/concatenate/concat:output:0*
T0*
_output_shapes
:21
/DQN_Noisy_Net_Categorical_Dueling/reshape/ShapeÈ
=DQN_Noisy_Net_Categorical_Dueling/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=DQN_Noisy_Net_Categorical_Dueling/reshape/strided_slice/stackÌ
?DQN_Noisy_Net_Categorical_Dueling/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?DQN_Noisy_Net_Categorical_Dueling/reshape/strided_slice/stack_1Ì
?DQN_Noisy_Net_Categorical_Dueling/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?DQN_Noisy_Net_Categorical_Dueling/reshape/strided_slice/stack_2Þ
7DQN_Noisy_Net_Categorical_Dueling/reshape/strided_sliceStridedSlice8DQN_Noisy_Net_Categorical_Dueling/reshape/Shape:output:0FDQN_Noisy_Net_Categorical_Dueling/reshape/strided_slice/stack:output:0HDQN_Noisy_Net_Categorical_Dueling/reshape/strided_slice/stack_1:output:0HDQN_Noisy_Net_Categorical_Dueling/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7DQN_Noisy_Net_Categorical_Dueling/reshape/strided_slice¸
9DQN_Noisy_Net_Categorical_Dueling/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2;
9DQN_Noisy_Net_Categorical_Dueling/reshape/Reshape/shape/1¸
9DQN_Noisy_Net_Categorical_Dueling/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :32;
9DQN_Noisy_Net_Categorical_Dueling/reshape/Reshape/shape/2ò
7DQN_Noisy_Net_Categorical_Dueling/reshape/Reshape/shapePack@DQN_Noisy_Net_Categorical_Dueling/reshape/strided_slice:output:0BDQN_Noisy_Net_Categorical_Dueling/reshape/Reshape/shape/1:output:0BDQN_Noisy_Net_Categorical_Dueling/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:29
7DQN_Noisy_Net_Categorical_Dueling/reshape/Reshape/shape¨
1DQN_Noisy_Net_Categorical_Dueling/reshape/ReshapeReshape=DQN_Noisy_Net_Categorical_Dueling/concatenate/concat:output:0@DQN_Noisy_Net_Categorical_Dueling/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ323
1DQN_Noisy_Net_Categorical_Dueling/reshape/Reshape©
IDQN_Noisy_Net_Categorical_Dueling/categorical_dense/MatMul/ReadVariableOpReadVariableOpRdqn_noisy_net_categorical_dueling_categorical_dense_matmul_readvariableop_resource*
_output_shapes

: 3*
dtype02K
IDQN_Noisy_Net_Categorical_Dueling/categorical_dense/MatMul/ReadVariableOpË
:DQN_Noisy_Net_Categorical_Dueling/categorical_dense/MatMulMatMulBDQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/Relu:activations:0QDQN_Noisy_Net_Categorical_Dueling/categorical_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32<
:DQN_Noisy_Net_Categorical_Dueling/categorical_dense/MatMul¨
JDQN_Noisy_Net_Categorical_Dueling/categorical_dense/BiasAdd/ReadVariableOpReadVariableOpSdqn_noisy_net_categorical_dueling_categorical_dense_biasadd_readvariableop_resource*
_output_shapes
:3*
dtype02L
JDQN_Noisy_Net_Categorical_Dueling/categorical_dense/BiasAdd/ReadVariableOpÑ
;DQN_Noisy_Net_Categorical_Dueling/categorical_dense/BiasAddBiasAddDDQN_Noisy_Net_Categorical_Dueling/categorical_dense/MatMul:product:0RDQN_Noisy_Net_Categorical_Dueling/categorical_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32=
;DQN_Noisy_Net_Categorical_Dueling/categorical_dense/BiasAdd´
7DQN_Noisy_Net_Categorical_Dueling/lambda/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :29
7DQN_Noisy_Net_Categorical_Dueling/lambda/ExpandDims/dim¶
3DQN_Noisy_Net_Categorical_Dueling/lambda/ExpandDims
ExpandDimsDDQN_Noisy_Net_Categorical_Dueling/categorical_dense/BiasAdd:output:0@DQN_Noisy_Net_Categorical_Dueling/lambda/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ325
3DQN_Noisy_Net_Categorical_Dueling/lambda/ExpandDimsÄ
?DQN_Noisy_Net_Categorical_Dueling/lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2A
?DQN_Noisy_Net_Categorical_Dueling/lambda/Mean/reduction_indices³
-DQN_Noisy_Net_Categorical_Dueling/lambda/MeanMean:DQN_Noisy_Net_Categorical_Dueling/reshape/Reshape:output:0HDQN_Noisy_Net_Categorical_Dueling/lambda/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*
	keep_dims(2/
-DQN_Noisy_Net_Categorical_Dueling/lambda/Mean
,DQN_Noisy_Net_Categorical_Dueling/lambda/subSub:DQN_Noisy_Net_Categorical_Dueling/reshape/Reshape:output:06DQN_Noisy_Net_Categorical_Dueling/lambda/Mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32.
,DQN_Noisy_Net_Categorical_Dueling/lambda/sub
,DQN_Noisy_Net_Categorical_Dueling/lambda/addAddV2<DQN_Noisy_Net_Categorical_Dueling/lambda/ExpandDims:output:00DQN_Noisy_Net_Categorical_Dueling/lambda/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32.
,DQN_Noisy_Net_Categorical_Dueling/lambda/addÓ
BDQN_Noisy_Net_Categorical_Dueling/activation/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2D
BDQN_Noisy_Net_Categorical_Dueling/activation/Max/reduction_indices±
0DQN_Noisy_Net_Categorical_Dueling/activation/MaxMax0DQN_Noisy_Net_Categorical_Dueling/lambda/add:z:0KDQN_Noisy_Net_Categorical_Dueling/activation/Max/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(22
0DQN_Noisy_Net_Categorical_Dueling/activation/Max
0DQN_Noisy_Net_Categorical_Dueling/activation/subSub0DQN_Noisy_Net_Categorical_Dueling/lambda/add:z:09DQN_Noisy_Net_Categorical_Dueling/activation/Max:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ322
0DQN_Noisy_Net_Categorical_Dueling/activation/sub×
0DQN_Noisy_Net_Categorical_Dueling/activation/ExpExp4DQN_Noisy_Net_Categorical_Dueling/activation/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ322
0DQN_Noisy_Net_Categorical_Dueling/activation/ExpÓ
BDQN_Noisy_Net_Categorical_Dueling/activation/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2D
BDQN_Noisy_Net_Categorical_Dueling/activation/Sum/reduction_indicesµ
0DQN_Noisy_Net_Categorical_Dueling/activation/SumSum4DQN_Noisy_Net_Categorical_Dueling/activation/Exp:y:0KDQN_Noisy_Net_Categorical_Dueling/activation/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(22
0DQN_Noisy_Net_Categorical_Dueling/activation/Sum
4DQN_Noisy_Net_Categorical_Dueling/activation/truedivRealDiv4DQN_Noisy_Net_Categorical_Dueling/activation/Exp:y:09DQN_Noisy_Net_Categorical_Dueling/activation/Sum:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ326
4DQN_Noisy_Net_Categorical_Dueling/activation/truedivé
IdentityIdentity8DQN_Noisy_Net_Categorical_Dueling/activation/truediv:z:0K^DQN_Noisy_Net_Categorical_Dueling/categorical_dense/BiasAdd/ReadVariableOpJ^DQN_Noisy_Net_Categorical_Dueling/categorical_dense/MatMul/ReadVariableOpE^DQN_Noisy_Net_Categorical_Dueling/noisy_dense/BiasAdd/ReadVariableOpD^DQN_Noisy_Net_Categorical_Dueling/noisy_dense/MatMul/ReadVariableOpG^DQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/BiasAdd/ReadVariableOpF^DQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/MatMul/ReadVariableOpG^DQN_Noisy_Net_Categorical_Dueling/noisy_dense_2/BiasAdd/ReadVariableOpF^DQN_Noisy_Net_Categorical_Dueling/noisy_dense_2/MatMul/ReadVariableOpG^DQN_Noisy_Net_Categorical_Dueling/noisy_dense_3/BiasAdd/ReadVariableOpF^DQN_Noisy_Net_Categorical_Dueling/noisy_dense_3/MatMul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::::2
JDQN_Noisy_Net_Categorical_Dueling/categorical_dense/BiasAdd/ReadVariableOpJDQN_Noisy_Net_Categorical_Dueling/categorical_dense/BiasAdd/ReadVariableOp2
IDQN_Noisy_Net_Categorical_Dueling/categorical_dense/MatMul/ReadVariableOpIDQN_Noisy_Net_Categorical_Dueling/categorical_dense/MatMul/ReadVariableOp2
DDQN_Noisy_Net_Categorical_Dueling/noisy_dense/BiasAdd/ReadVariableOpDDQN_Noisy_Net_Categorical_Dueling/noisy_dense/BiasAdd/ReadVariableOp2
CDQN_Noisy_Net_Categorical_Dueling/noisy_dense/MatMul/ReadVariableOpCDQN_Noisy_Net_Categorical_Dueling/noisy_dense/MatMul/ReadVariableOp2
FDQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/BiasAdd/ReadVariableOpFDQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/BiasAdd/ReadVariableOp2
EDQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/MatMul/ReadVariableOpEDQN_Noisy_Net_Categorical_Dueling/noisy_dense_1/MatMul/ReadVariableOp2
FDQN_Noisy_Net_Categorical_Dueling/noisy_dense_2/BiasAdd/ReadVariableOpFDQN_Noisy_Net_Categorical_Dueling/noisy_dense_2/BiasAdd/ReadVariableOp2
EDQN_Noisy_Net_Categorical_Dueling/noisy_dense_2/MatMul/ReadVariableOpEDQN_Noisy_Net_Categorical_Dueling/noisy_dense_2/MatMul/ReadVariableOp2
FDQN_Noisy_Net_Categorical_Dueling/noisy_dense_3/BiasAdd/ReadVariableOpFDQN_Noisy_Net_Categorical_Dueling/noisy_dense_3/BiasAdd/ReadVariableOp2
EDQN_Noisy_Net_Categorical_Dueling/noisy_dense_3/MatMul/ReadVariableOpEDQN_Noisy_Net_Categorical_Dueling/noisy_dense_3/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
	
d
H__inference_activation_layer_call_and_return_conditional_losses_21168506

inputs
identityy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Max/reduction_indices
MaxMaxinputsMax/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Max]
subSubinputsMax:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
subP
ExpExpsub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Sum/reduction_indices
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Sumj
truedivRealDivExp:y:0Sum:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32	
truedivc
IdentityIdentitytruediv:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
 
_user_specified_nameinputs
õ	
ä
K__inference_noisy_dense_1_layer_call_and_return_conditional_losses_21167273

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
Á
0__inference_noisy_dense_2_layer_call_fn_21168309

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_noisy_dense_2_layer_call_and_return_conditional_losses_211673342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ :: 3:::3:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:$ 

_output_shapes

: 3: 

_output_shapes
:3
ª
I
-__inference_activation_layer_call_fn_21168511

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_211676122
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
 
_user_specified_nameinputs
¹
Á
0__inference_noisy_dense_3_layer_call_fn_21168365

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_noisy_dense_3_layer_call_and_return_conditional_losses_211674052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ32

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ :: 3:::3:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:$ 

_output_shapes

: 3: 

_output_shapes
:3"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*±
serving_default
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿB

activation4
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ3tensorflow/serving/predict:©Æ
ÂP
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
layer_with_weights-4
layer-7
	layer-8

layer-9
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+­&call_and_return_all_conditional_losses
®_default_save_signature
¯__call__"ðL
_tf_keras_networkÔL{"class_name": "Functional", "name": "DQN_Noisy_Net_Categorical_Dueling", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "DQN_Noisy_Net_Categorical_Dueling", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "NoisyDense", "config": {"name": "noisy_dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.5, "maxval": 0.5, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.25}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.125}}}, "name": "noisy_dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "NoisyDense", "config": {"name": "noisy_dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.25, "maxval": 0.25, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.125}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.08838834764831843}}}, "name": "noisy_dense_1", "inbound_nodes": [[["noisy_dense", 0, 0, {}]]]}, {"class_name": "NoisyDense", "config": {"name": "noisy_dense_2", "trainable": true, "dtype": "float32", "units": 51, "activation": "linear", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.17677669529663687, "maxval": 0.17677669529663687, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.08838834764831843}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.07001400420140048}}}, "name": "noisy_dense_2", "inbound_nodes": [[["noisy_dense_1", 0, 0, {}]]]}, {"class_name": "NoisyDense", "config": {"name": "noisy_dense_3", "trainable": true, "dtype": "float32", "units": 51, "activation": "linear", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.17677669529663687, "maxval": 0.17677669529663687, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.08838834764831843}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.07001400420140048}}}, "name": "noisy_dense_3", "inbound_nodes": [[["noisy_dense_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["noisy_dense_2", 0, 0, {}], ["noisy_dense_3", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 51]}}, "name": "reshape", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "NoisyDense", "config": {"name": "categorical_dense", "trainable": true, "dtype": "float32", "units": 51, "activation": "linear", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.17677669529663687, "maxval": 0.17677669529663687, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.08838834764831843}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.07001400420140048}}}, "name": "categorical_dense", "inbound_nodes": [[["noisy_dense_1", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAMAAAAHAAAAUwAAAHMoAAAAfABcAn0BfQJ0AKABfAJkAaECfAF0AGoCfAFk\nAWQCZAONAxgAFwBTAKkETukBAAAAVCkC2gRheGlz2ghrZWVwZGltcykD2gFL2gtleHBhbmRfZGlt\nc9oEbWVhbqkD2gFz2gFh2gF2qQByDAAAAPo+L2hvbWUvcGhpbGlwcGV0dXJuZXIvc291cmNlcy9J\nTkY4MjI1LVBST0pFVC9yYWluYm93L3JhaW5ib3cucHnaCGF2Z19kdWVshAEAAHMEAAAAAAEIAQ==\n", null, null]}, "function_type": "lambda", "module": "rainbow.rainbow", "output_shape": {"class_name": "__tuple__", "items": [2, 51]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["reshape", 0, 0, {}], ["categorical_dense", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "softmax"}, "name": "activation", "inbound_nodes": [[["lambda", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["activation", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 4]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "DQN_Noisy_Net_Categorical_Dueling", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "NoisyDense", "config": {"name": "noisy_dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.5, "maxval": 0.5, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.25}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.125}}}, "name": "noisy_dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "NoisyDense", "config": {"name": "noisy_dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.25, "maxval": 0.25, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.125}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.08838834764831843}}}, "name": "noisy_dense_1", "inbound_nodes": [[["noisy_dense", 0, 0, {}]]]}, {"class_name": "NoisyDense", "config": {"name": "noisy_dense_2", "trainable": true, "dtype": "float32", "units": 51, "activation": "linear", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.17677669529663687, "maxval": 0.17677669529663687, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.08838834764831843}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.07001400420140048}}}, "name": "noisy_dense_2", "inbound_nodes": [[["noisy_dense_1", 0, 0, {}]]]}, {"class_name": "NoisyDense", "config": {"name": "noisy_dense_3", "trainable": true, "dtype": "float32", "units": 51, "activation": "linear", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.17677669529663687, "maxval": 0.17677669529663687, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.08838834764831843}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.07001400420140048}}}, "name": "noisy_dense_3", "inbound_nodes": [[["noisy_dense_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["noisy_dense_2", 0, 0, {}], ["noisy_dense_3", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 51]}}, "name": "reshape", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "NoisyDense", "config": {"name": "categorical_dense", "trainable": true, "dtype": "float32", "units": 51, "activation": "linear", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.17677669529663687, "maxval": 0.17677669529663687, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.08838834764831843}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.07001400420140048}}}, "name": "categorical_dense", "inbound_nodes": [[["noisy_dense_1", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAMAAAAHAAAAUwAAAHMoAAAAfABcAn0BfQJ0AKABfAJkAaECfAF0AGoCfAFk\nAWQCZAONAxgAFwBTAKkETukBAAAAVCkC2gRheGlz2ghrZWVwZGltcykD2gFL2gtleHBhbmRfZGlt\nc9oEbWVhbqkD2gFz2gFh2gF2qQByDAAAAPo+L2hvbWUvcGhpbGlwcGV0dXJuZXIvc291cmNlcy9J\nTkY4MjI1LVBST0pFVC9yYWluYm93L3JhaW5ib3cucHnaCGF2Z19kdWVshAEAAHMEAAAAAAEIAQ==\n", null, null]}, "function_type": "lambda", "module": "rainbow.rainbow", "output_shape": {"class_name": "__tuple__", "items": [2, 51]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["reshape", 0, 0, {}], ["categorical_dense", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "softmax"}, "name": "activation", "inbound_nodes": [[["lambda", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["activation", 0, 0]]}}, "training_config": {"loss": "modified_KL_Divergence", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
é"æ
_tf_keras_input_layerÆ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}

w_mu
w_sigma
b_mu
b_sigma

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+°&call_and_return_all_conditional_losses
±__call__"­
_tf_keras_layer{"class_name": "NoisyDense", "name": "noisy_dense", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "noisy_dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.5, "maxval": 0.5, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.25}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.125}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}

w_mu
w_sigma
b_mu
b_sigma

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
+²&call_and_return_all_conditional_losses
³__call__"Ã
_tf_keras_layer©{"class_name": "NoisyDense", "name": "noisy_dense_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "noisy_dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.25, "maxval": 0.25, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.125}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.08838834764831843}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
Æ
!w_mu
"w_sigma
#b_mu
$b_sigma

!kernel
#bias
%trainable_variables
&	variables
'regularization_losses
(	keras_api
+´&call_and_return_all_conditional_losses
µ__call__"ñ
_tf_keras_layer×{"class_name": "NoisyDense", "name": "noisy_dense_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "noisy_dense_2", "trainable": true, "dtype": "float32", "units": 51, "activation": "linear", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.17677669529663687, "maxval": 0.17677669529663687, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.08838834764831843}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.07001400420140048}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
Æ
)w_mu
*w_sigma
+b_mu
,b_sigma

)kernel
+bias
-trainable_variables
.	variables
/regularization_losses
0	keras_api
+¶&call_and_return_all_conditional_losses
·__call__"ñ
_tf_keras_layer×{"class_name": "NoisyDense", "name": "noisy_dense_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "noisy_dense_3", "trainable": true, "dtype": "float32", "units": 51, "activation": "linear", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.17677669529663687, "maxval": 0.17677669529663687, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.08838834764831843}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.07001400420140048}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
Ë
1trainable_variables
2	variables
3regularization_losses
4	keras_api
+¸&call_and_return_all_conditional_losses
¹__call__"º
_tf_keras_layer {"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 51]}, {"class_name": "TensorShape", "items": [null, 51]}]}
ó
5trainable_variables
6	variables
7regularization_losses
8	keras_api
+º&call_and_return_all_conditional_losses
»__call__"â
_tf_keras_layerÈ{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 51]}}}
Î
9w_mu
:w_sigma
;b_mu
<b_sigma

9kernel
;bias
=trainable_variables
>	variables
?regularization_losses
@	keras_api
+¼&call_and_return_all_conditional_losses
½__call__"ù
_tf_keras_layerß{"class_name": "NoisyDense", "name": "categorical_dense", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "categorical_dense", "trainable": true, "dtype": "float32", "units": 51, "activation": "linear", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.17677669529663687, "maxval": 0.17677669529663687, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.08838834764831843}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.07001400420140048}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
í
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
+¾&call_and_return_all_conditional_losses
¿__call__"Ü
_tf_keras_layerÂ{"class_name": "Lambda", "name": "lambda", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAMAAAAHAAAAUwAAAHMoAAAAfABcAn0BfQJ0AKABfAJkAaECfAF0AGoCfAFk\nAWQCZAONAxgAFwBTAKkETukBAAAAVCkC2gRheGlz2ghrZWVwZGltcykD2gFL2gtleHBhbmRfZGlt\nc9oEbWVhbqkD2gFz2gFh2gF2qQByDAAAAPo+L2hvbWUvcGhpbGlwcGV0dXJuZXIvc291cmNlcy9J\nTkY4MjI1LVBST0pFVC9yYWluYm93L3JhaW5ib3cucHnaCGF2Z19kdWVshAEAAHMEAAAAAAEIAQ==\n", null, null]}, "function_type": "lambda", "module": "rainbow.rainbow", "output_shape": {"class_name": "__tuple__", "items": [2, 51]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
Ö
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
+À&call_and_return_all_conditional_losses
Á__call__"Å
_tf_keras_layer«{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "softmax"}}
ã

Ibeta_1

Jbeta_2
	Kdecay
Llearning_rate
Mitermmmmmmmm!m"m#m$m)m*m+m,m9m:m;m<mvvvvvvvv !v¡"v¢#v£$v¤)v¥*v¦+v§,v¨9v©:vª;v«<v¬"
	optimizer
¶
0
1
2
3
4
5
6
7
!8
"9
#10
$11
)12
*13
+14
,15
916
:17
;18
<19"
trackable_list_wrapper
¶
0
1
2
3
4
5
6
7
!8
"9
#10
$11
)12
*13
+14
,15
916
:17
;18
<19"
trackable_list_wrapper
 "
trackable_list_wrapper
Î

Nlayers
trainable_variables
	variables
Olayer_regularization_losses
regularization_losses
Player_metrics
Qmetrics
Rnon_trainable_variables
¯__call__
®_default_save_signature
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
-
Âserving_default"
signature_map
": 2noisy_dense/w_mu
%:#2noisy_dense/w_sigma
:2noisy_dense/b_mu
!:2noisy_dense/b_sigma
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
°

Slayers
trainable_variables
	variables
Tlayer_regularization_losses
regularization_losses
Ulayer_metrics
Vmetrics
Wnon_trainable_variables
±__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
$:" 2noisy_dense_1/w_mu
':% 2noisy_dense_1/w_sigma
 : 2noisy_dense_1/b_mu
#:! 2noisy_dense_1/b_sigma
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
°

Xlayers
trainable_variables
	variables
Ylayer_regularization_losses
regularization_losses
Zlayer_metrics
[metrics
\non_trainable_variables
³__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
$:" 32noisy_dense_2/w_mu
':% 32noisy_dense_2/w_sigma
 :32noisy_dense_2/b_mu
#:!32noisy_dense_2/b_sigma
<
!0
"1
#2
$3"
trackable_list_wrapper
<
!0
"1
#2
$3"
trackable_list_wrapper
 "
trackable_list_wrapper
°

]layers
%trainable_variables
&	variables
^layer_regularization_losses
'regularization_losses
_layer_metrics
`metrics
anon_trainable_variables
µ__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
$:" 32noisy_dense_3/w_mu
':% 32noisy_dense_3/w_sigma
 :32noisy_dense_3/b_mu
#:!32noisy_dense_3/b_sigma
<
)0
*1
+2
,3"
trackable_list_wrapper
<
)0
*1
+2
,3"
trackable_list_wrapper
 "
trackable_list_wrapper
°

blayers
-trainable_variables
.	variables
clayer_regularization_losses
/regularization_losses
dlayer_metrics
emetrics
fnon_trainable_variables
·__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

glayers
1trainable_variables
2	variables
hlayer_regularization_losses
3regularization_losses
ilayer_metrics
jmetrics
knon_trainable_variables
¹__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

llayers
5trainable_variables
6	variables
mlayer_regularization_losses
7regularization_losses
nlayer_metrics
ometrics
pnon_trainable_variables
»__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
(:& 32categorical_dense/w_mu
+:) 32categorical_dense/w_sigma
$:"32categorical_dense/b_mu
':%32categorical_dense/b_sigma
<
90
:1
;2
<3"
trackable_list_wrapper
<
90
:1
;2
<3"
trackable_list_wrapper
 "
trackable_list_wrapper
°

qlayers
=trainable_variables
>	variables
rlayer_regularization_losses
?regularization_losses
slayer_metrics
tmetrics
unon_trainable_variables
½__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

vlayers
Atrainable_variables
B	variables
wlayer_regularization_losses
Cregularization_losses
xlayer_metrics
ymetrics
znon_trainable_variables
¿__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

{layers
Etrainable_variables
F	variables
|layer_regularization_losses
Gregularization_losses
}layer_metrics
~metrics
non_trainable_variables
Á__call__
+À&call_and_return_all_conditional_losses
'À"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
0"
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
¿

total

count
	variables
	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
':%2Adam/noisy_dense/w_mu/m
*:(2Adam/noisy_dense/w_sigma/m
#:!2Adam/noisy_dense/b_mu/m
&:$2Adam/noisy_dense/b_sigma/m
):' 2Adam/noisy_dense_1/w_mu/m
,:* 2Adam/noisy_dense_1/w_sigma/m
%:# 2Adam/noisy_dense_1/b_mu/m
(:& 2Adam/noisy_dense_1/b_sigma/m
):' 32Adam/noisy_dense_2/w_mu/m
,:* 32Adam/noisy_dense_2/w_sigma/m
%:#32Adam/noisy_dense_2/b_mu/m
(:&32Adam/noisy_dense_2/b_sigma/m
):' 32Adam/noisy_dense_3/w_mu/m
,:* 32Adam/noisy_dense_3/w_sigma/m
%:#32Adam/noisy_dense_3/b_mu/m
(:&32Adam/noisy_dense_3/b_sigma/m
-:+ 32Adam/categorical_dense/w_mu/m
0:. 32 Adam/categorical_dense/w_sigma/m
):'32Adam/categorical_dense/b_mu/m
,:*32 Adam/categorical_dense/b_sigma/m
':%2Adam/noisy_dense/w_mu/v
*:(2Adam/noisy_dense/w_sigma/v
#:!2Adam/noisy_dense/b_mu/v
&:$2Adam/noisy_dense/b_sigma/v
):' 2Adam/noisy_dense_1/w_mu/v
,:* 2Adam/noisy_dense_1/w_sigma/v
%:# 2Adam/noisy_dense_1/b_mu/v
(:& 2Adam/noisy_dense_1/b_sigma/v
):' 32Adam/noisy_dense_2/w_mu/v
,:* 32Adam/noisy_dense_2/w_sigma/v
%:#32Adam/noisy_dense_2/b_mu/v
(:&32Adam/noisy_dense_2/b_sigma/v
):' 32Adam/noisy_dense_3/w_mu/v
,:* 32Adam/noisy_dense_3/w_sigma/v
%:#32Adam/noisy_dense_3/b_mu/v
(:&32Adam/noisy_dense_3/b_sigma/v
-:+ 32Adam/categorical_dense/w_mu/v
0:. 32 Adam/categorical_dense/w_sigma/v
):'32Adam/categorical_dense/b_mu/v
,:*32 Adam/categorical_dense/b_sigma/v
Ê2Ç
___inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_and_return_conditional_losses_21167654
___inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_and_return_conditional_losses_21167621
___inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_and_return_conditional_losses_21167996
___inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_and_return_conditional_losses_21168056À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
á2Þ
#__inference__wrapped_model_21167164¶
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
Þ2Û
D__inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_fn_21168121
D__inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_fn_21167793
D__inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_fn_21167851
D__inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_fn_21168146À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ð2Í
I__inference_noisy_dense_layer_call_and_return_conditional_losses_21168167
I__inference_noisy_dense_layer_call_and_return_conditional_losses_21168178´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
.__inference_noisy_dense_layer_call_fn_21168195
.__inference_noisy_dense_layer_call_fn_21168204´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
K__inference_noisy_dense_1_layer_call_and_return_conditional_losses_21168225
K__inference_noisy_dense_1_layer_call_and_return_conditional_losses_21168236´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
0__inference_noisy_dense_1_layer_call_fn_21168253
0__inference_noisy_dense_1_layer_call_fn_21168262´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
K__inference_noisy_dense_2_layer_call_and_return_conditional_losses_21168282
K__inference_noisy_dense_2_layer_call_and_return_conditional_losses_21168292´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
0__inference_noisy_dense_2_layer_call_fn_21168309
0__inference_noisy_dense_2_layer_call_fn_21168318´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
K__inference_noisy_dense_3_layer_call_and_return_conditional_losses_21168348
K__inference_noisy_dense_3_layer_call_and_return_conditional_losses_21168338´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
0__inference_noisy_dense_3_layer_call_fn_21168374
0__inference_noisy_dense_3_layer_call_fn_21168365´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ó2ð
I__inference_concatenate_layer_call_and_return_conditional_losses_21168381¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_concatenate_layer_call_fn_21168387¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_reshape_layer_call_and_return_conditional_losses_21168400¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_reshape_layer_call_fn_21168405¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ü2Ù
O__inference_categorical_dense_layer_call_and_return_conditional_losses_21168425
O__inference_categorical_dense_layer_call_and_return_conditional_losses_21168435´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¦2£
4__inference_categorical_dense_layer_call_fn_21168452
4__inference_categorical_dense_layer_call_fn_21168461´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
D__inference_lambda_layer_call_and_return_conditional_losses_21168472
D__inference_lambda_layer_call_and_return_conditional_losses_21168483À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
)__inference_lambda_layer_call_fn_21168489
)__inference_lambda_layer_call_fn_21168495À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
H__inference_activation_layer_call_and_return_conditional_losses_21168506¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_activation_layer_call_fn_21168511¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÍBÊ
&__inference_signature_wrapper_21167886input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5
J	
Const_6
J	
Const_7
J	
Const_8
J	
Const_9ó
___inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_and_return_conditional_losses_21167621(ÃÄÅÆ"Ç!$È#*É),Ê+:Ë9<Ì;8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ3
 Ô
___inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_and_return_conditional_losses_21167654q
!#)+9;8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ3
 ò
___inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_and_return_conditional_losses_21167996(ÃÄÅÆ"Ç!$È#*É),Ê+:Ë9<Ì;7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ3
 Ó
___inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_and_return_conditional_losses_21168056p
!#)+9;7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ3
 Ë
D__inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_fn_21167793(ÃÄÅÆ"Ç!$È#*É),Ê+:Ë9<Ì;8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ3¬
D__inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_fn_21167851d
!#)+9;8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ3Ê
D__inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_fn_21168121(ÃÄÅÆ"Ç!$È#*É),Ê+:Ë9<Ì;7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ3«
D__inference_DQN_Noisy_Net_Categorical_Dueling_layer_call_fn_21168146c
!#)+9;7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ3¢
#__inference__wrapped_model_21167164{
!#)+9;0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª ";ª8
6

activation(%

activationÿÿÿÿÿÿÿÿÿ3¬
H__inference_activation_layer_call_and_return_conditional_losses_21168506`3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ3
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ3
 
-__inference_activation_layer_call_fn_21168511S3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ3
ª "ÿÿÿÿÿÿÿÿÿ3¹
O__inference_categorical_dense_layer_call_and_return_conditional_losses_21168425f:Ë9<Ì;3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ3
 ³
O__inference_categorical_dense_layer_call_and_return_conditional_losses_21168435`9;3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ3
 
4__inference_categorical_dense_layer_call_fn_21168452Y:Ë9<Ì;3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "ÿÿÿÿÿÿÿÿÿ3
4__inference_categorical_dense_layer_call_fn_21168461S9;3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "ÿÿÿÿÿÿÿÿÿ3Ñ
I__inference_concatenate_layer_call_and_return_conditional_losses_21168381Z¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ3
"
inputs/1ÿÿÿÿÿÿÿÿÿ3
ª "%¢"

0ÿÿÿÿÿÿÿÿÿf
 ¨
.__inference_concatenate_layer_call_fn_21168387vZ¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ3
"
inputs/1ÿÿÿÿÿÿÿÿÿ3
ª "ÿÿÿÿÿÿÿÿÿfÜ
D__inference_lambda_layer_call_and_return_conditional_losses_21168472f¢c
\¢Y
OL
&#
inputs/0ÿÿÿÿÿÿÿÿÿ3
"
inputs/1ÿÿÿÿÿÿÿÿÿ3

 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ3
 Ü
D__inference_lambda_layer_call_and_return_conditional_losses_21168483f¢c
\¢Y
OL
&#
inputs/0ÿÿÿÿÿÿÿÿÿ3
"
inputs/1ÿÿÿÿÿÿÿÿÿ3

 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ3
 ´
)__inference_lambda_layer_call_fn_21168489f¢c
\¢Y
OL
&#
inputs/0ÿÿÿÿÿÿÿÿÿ3
"
inputs/1ÿÿÿÿÿÿÿÿÿ3

 
p
ª "ÿÿÿÿÿÿÿÿÿ3´
)__inference_lambda_layer_call_fn_21168495f¢c
\¢Y
OL
&#
inputs/0ÿÿÿÿÿÿÿÿÿ3
"
inputs/1ÿÿÿÿÿÿÿÿÿ3

 
p 
ª "ÿÿÿÿÿÿÿÿÿ3µ
K__inference_noisy_dense_1_layer_call_and_return_conditional_losses_21168225fÅÆ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ¯
K__inference_noisy_dense_1_layer_call_and_return_conditional_losses_21168236`3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
0__inference_noisy_dense_1_layer_call_fn_21168253YÅÆ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ 
0__inference_noisy_dense_1_layer_call_fn_21168262S3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ µ
K__inference_noisy_dense_2_layer_call_and_return_conditional_losses_21168282f"Ç!$È#3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ3
 ¯
K__inference_noisy_dense_2_layer_call_and_return_conditional_losses_21168292`!#3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ3
 
0__inference_noisy_dense_2_layer_call_fn_21168309Y"Ç!$È#3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "ÿÿÿÿÿÿÿÿÿ3
0__inference_noisy_dense_2_layer_call_fn_21168318S!#3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "ÿÿÿÿÿÿÿÿÿ3µ
K__inference_noisy_dense_3_layer_call_and_return_conditional_losses_21168338f*É),Ê+3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ3
 ¯
K__inference_noisy_dense_3_layer_call_and_return_conditional_losses_21168348`)+3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ3
 
0__inference_noisy_dense_3_layer_call_fn_21168365Y*É),Ê+3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "ÿÿÿÿÿÿÿÿÿ3
0__inference_noisy_dense_3_layer_call_fn_21168374S)+3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "ÿÿÿÿÿÿÿÿÿ3³
I__inference_noisy_dense_layer_call_and_return_conditional_losses_21168167fÃÄ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ­
I__inference_noisy_dense_layer_call_and_return_conditional_losses_21168178`3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_noisy_dense_layer_call_fn_21168195YÃÄ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_noisy_dense_layer_call_fn_21168204S3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_reshape_layer_call_and_return_conditional_losses_21168400\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿf
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ3
 }
*__inference_reshape_layer_call_fn_21168405O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿf
ª "ÿÿÿÿÿÿÿÿÿ3±
&__inference_signature_wrapper_21167886
!#)+9;;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ";ª8
6

activation(%

activationÿÿÿÿÿÿÿÿÿ3