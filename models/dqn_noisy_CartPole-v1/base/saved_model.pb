??
??
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
8
Const
output"dtype"
valuetensor"
dtypetype
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
E
Relu
features"T
activations"T"
Ttype:
2	
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
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
?
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
?
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
?
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
?
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
?
noisy_dense_2/w_muVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *#
shared_namenoisy_dense_2/w_mu
y
&noisy_dense_2/w_mu/Read/ReadVariableOpReadVariableOpnoisy_dense_2/w_mu*
_output_shapes

: *
dtype0
?
noisy_dense_2/w_sigmaVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_namenoisy_dense_2/w_sigma

)noisy_dense_2/w_sigma/Read/ReadVariableOpReadVariableOpnoisy_dense_2/w_sigma*
_output_shapes

: *
dtype0
|
noisy_dense_2/b_muVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namenoisy_dense_2/b_mu
u
&noisy_dense_2/b_mu/Read/ReadVariableOpReadVariableOpnoisy_dense_2/b_mu*
_output_shapes
:*
dtype0
?
noisy_dense_2/b_sigmaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namenoisy_dense_2/b_sigma
{
)noisy_dense_2/b_sigma/Read/ReadVariableOpReadVariableOpnoisy_dense_2/b_sigma*
_output_shapes
:*
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
Adam/noisy_dense/w_mu/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/noisy_dense/w_mu/m
?
+Adam/noisy_dense/w_mu/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense/w_mu/m*
_output_shapes

:*
dtype0
?
Adam/noisy_dense/w_sigma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/noisy_dense/w_sigma/m
?
.Adam/noisy_dense/w_sigma/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense/w_sigma/m*
_output_shapes

:*
dtype0
?
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
?
Adam/noisy_dense/b_sigma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/noisy_dense/b_sigma/m
?
.Adam/noisy_dense/b_sigma/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense/b_sigma/m*
_output_shapes
:*
dtype0
?
Adam/noisy_dense_1/w_mu/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: **
shared_nameAdam/noisy_dense_1/w_mu/m
?
-Adam/noisy_dense_1/w_mu/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_1/w_mu/m*
_output_shapes

: *
dtype0
?
Adam/noisy_dense_1/w_sigma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *-
shared_nameAdam/noisy_dense_1/w_sigma/m
?
0Adam/noisy_dense_1/w_sigma/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_1/w_sigma/m*
_output_shapes

: *
dtype0
?
Adam/noisy_dense_1/b_mu/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/noisy_dense_1/b_mu/m
?
-Adam/noisy_dense_1/b_mu/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_1/b_mu/m*
_output_shapes
: *
dtype0
?
Adam/noisy_dense_1/b_sigma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/noisy_dense_1/b_sigma/m
?
0Adam/noisy_dense_1/b_sigma/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_1/b_sigma/m*
_output_shapes
: *
dtype0
?
Adam/noisy_dense_2/w_mu/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: **
shared_nameAdam/noisy_dense_2/w_mu/m
?
-Adam/noisy_dense_2/w_mu/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_2/w_mu/m*
_output_shapes

: *
dtype0
?
Adam/noisy_dense_2/w_sigma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *-
shared_nameAdam/noisy_dense_2/w_sigma/m
?
0Adam/noisy_dense_2/w_sigma/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_2/w_sigma/m*
_output_shapes

: *
dtype0
?
Adam/noisy_dense_2/b_mu/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/noisy_dense_2/b_mu/m
?
-Adam/noisy_dense_2/b_mu/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_2/b_mu/m*
_output_shapes
:*
dtype0
?
Adam/noisy_dense_2/b_sigma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/noisy_dense_2/b_sigma/m
?
0Adam/noisy_dense_2/b_sigma/m/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_2/b_sigma/m*
_output_shapes
:*
dtype0
?
Adam/noisy_dense/w_mu/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/noisy_dense/w_mu/v
?
+Adam/noisy_dense/w_mu/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense/w_mu/v*
_output_shapes

:*
dtype0
?
Adam/noisy_dense/w_sigma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/noisy_dense/w_sigma/v
?
.Adam/noisy_dense/w_sigma/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense/w_sigma/v*
_output_shapes

:*
dtype0
?
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
?
Adam/noisy_dense/b_sigma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/noisy_dense/b_sigma/v
?
.Adam/noisy_dense/b_sigma/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense/b_sigma/v*
_output_shapes
:*
dtype0
?
Adam/noisy_dense_1/w_mu/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: **
shared_nameAdam/noisy_dense_1/w_mu/v
?
-Adam/noisy_dense_1/w_mu/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_1/w_mu/v*
_output_shapes

: *
dtype0
?
Adam/noisy_dense_1/w_sigma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *-
shared_nameAdam/noisy_dense_1/w_sigma/v
?
0Adam/noisy_dense_1/w_sigma/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_1/w_sigma/v*
_output_shapes

: *
dtype0
?
Adam/noisy_dense_1/b_mu/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/noisy_dense_1/b_mu/v
?
-Adam/noisy_dense_1/b_mu/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_1/b_mu/v*
_output_shapes
: *
dtype0
?
Adam/noisy_dense_1/b_sigma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/noisy_dense_1/b_sigma/v
?
0Adam/noisy_dense_1/b_sigma/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_1/b_sigma/v*
_output_shapes
: *
dtype0
?
Adam/noisy_dense_2/w_mu/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: **
shared_nameAdam/noisy_dense_2/w_mu/v
?
-Adam/noisy_dense_2/w_mu/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_2/w_mu/v*
_output_shapes

: *
dtype0
?
Adam/noisy_dense_2/w_sigma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *-
shared_nameAdam/noisy_dense_2/w_sigma/v
?
0Adam/noisy_dense_2/w_sigma/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_2/w_sigma/v*
_output_shapes

: *
dtype0
?
Adam/noisy_dense_2/b_mu/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/noisy_dense_2/b_mu/v
?
-Adam/noisy_dense_2/b_mu/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_2/b_mu/v*
_output_shapes
:*
dtype0
?
Adam/noisy_dense_2/b_sigma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/noisy_dense_2/b_sigma/v
?
0Adam/noisy_dense_2/b_sigma/v/Read/ReadVariableOpReadVariableOpAdam/noisy_dense_2/b_sigma/v*
_output_shapes
:*
dtype0
?
ConstConst*
_output_shapes

:*
dtype0*?
value?B?"??!??{!?>6:$?˾???>???????????*%?%<
??`2????>??D>J`?>Y??=&?	???𾯢+?T?f????s??̆?Ƒ9??J?zh?_B?&?z???/??Z?>?R/??G>+?A?GUc??&??)???Ն?
?s?B????P???{??2??????`???7Υ?̵?Ѣ??翀>wڶ?Ɯ6??@??s>??D?X???C?Y????ӌ??OJ???????#X??>0??v??>b??#?N>?ᒿ
?
Const_1Const*
_output_shapes
:*
dtype0*U
valueLBJ"@?|*?z5s?t????<J??6?W??	z??r?<??b??????r???O?x??ľ7ox??A?!??
?
Const_2Const*
_output_shapes

: *
dtype0*?
value?B? "?6RҾ{^Q>h¢>s?ϾN1 ??*e??????i??Ő?><???u8??8???Dn?>ƌG??7????y???=??;?r>??>G?ʾ?????[0>?:w?E?>?A??p????>?]?=
߆=?V?=?>????$/????Э?9f?>????Q???m???H??????e?4???|1???&?p ????P>???C???i?J?9???????4??$z?[?N?a????!"?^I?Y??aw????a????Ej??Ϥ??$????ࢿb?Ⱦj?3?p??u???Y?????ȿW?b?p?????^?? p???C?Xk?>????4>???>bk??`2
?ٻA?I?????<??1??V??>?_S>䕓>??[??3??????xh?~p??D?>I?#?I??䰰?g;???Ƕ?N?C?۔[??Hv?o????Z??o2>????ui??OX-????א??????????0?z*??lv
??+??ʉ??Us?K?@?ˀ???H??o??????k?̥???ҹ>?&?j$???Q?? ????????F??^???y?'??^?d5>?????p??v?/??g??o???????????/3?QS??????.??׋?c?v???C?????K?6ϣ?b#??}?)㡿ůǾ?|2????ɲ??0????Vǿ޿U?{yo?_L???k?&?n?;?B??i?>/Ŕ??=?Y??>????B??W[	?e?@?:Q????~?;??F?????>R>???>?1Z?l?>?Hz?ܐ¾?x?>x>>???>?~?>????n???D	?>7Ƿ>? ξŋn>?(?>?W?=?Y???V?>????>??s?>C>?I?R??œ>?????g>???>X?澎?˽Q:???$Ὦr????/?{??>|???{-?S?U??E???; ?4?N?x?8?ܝU?J??NP ????p?????????н?00>m???>1??>?G)?7?J??1?>5Yξ7?)??ҡ?|?Ⱦ
!??1>Y#?= 2>???>e{A?.???o??76????k>???>??0???c?|?K?nrk??w?>5m??????>Z??Z??=*2B???/?4K߾?8??B?:?A?_??<???o?>??:?2\?>???>$1????V%??~B-?? ?c???[???mE??Y=??i>???>??.??ca???I??'i???>????iɵ>7?????=aN@?6.??ݾ?̐?s?8?[^]?????9?>?,9?ܟ?>?]?>??/?F3?'???ݒ+?6?????1U&??M??? ??E?˾?6?j???w?Ŀ???gT˿sZ??Dt?????/??trs??F????>??????@?g??>? ?????^?YiD??|?????M???H???X?>?KV>??>Y?^???K?Ưʾ??? 8I??3x>|??>??9??o?i~V??w?C????`?&?L.?>?S????=\L?L?8????????XD?]>k??????V?>%?D?뱻>??>L?:???$?	???S6??????=?????#????;?Ihg>r??>?F-?O_???G?g?E??>?
?ʡ???>uJ
????=?>?g,??۾?v???7?`S[?"-???$?>hw7????>iW?>?%.?????v????)?-???????
T?N?[?N?????]??񖁿/??????uì??@9?u?O???h?=????N???(??~?>???#?-??>?爿?????>??&?6?? ????"??=?? f>?6>?C~>3=?W?ؾ??W>=ۧ>?\־5?/Xl???ž))????>
???????????Ա>??M???N׀?=??=W?ľ)Tz>?#>A+Ѿ????e?5>??~????>q?G??Wx????>d??=J?=?<?=v?>c??>]?u?? ?????>p>?w?>?J?>J,????,?B?>?i?>9Zʾu-j>7γ>???=????g(?>?l??Ї:??>\????N???>????
?c>!K?>m????ǽzF??hݽ?a??
?
Const_3Const*
_output_shapes
: *
dtype0*?
value?B? "??;??X????c?????-{?>?m ??d??63????+???@?=>W?#kq????ɄV?V?.>Ǔ???????)???޾:???K???????-??L????P?(??????n???<?~؃??D?
?
Const_4Const*
_output_shapes

: *
dtype0*?
value?B? "?? ??.a??%????_??ǁ??S]?c?
?[??w-???>U?{??xV???T?e?5?3Б?}?x>?Ŷ??ٛ>+2/??c??0????o?;?j??.????i?+??>-پ7?c??GB?=:|>W?߄>*?b?'?0?s??aw^?l?=????????>Um?>??t?????/?>#^-????<k?E???C?>3ž)???s??(???y?j???O??/1?`????ﾈ??>?lҾe:?>.%???+?|???
X
Const_5Const*
_output_shapes
:*
dtype0*
valueB"??B?,-&?

NoOpNoOp
?8
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?7
value?7B?7 B?7
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
regularization_losses
trainable_variables
	variables
		keras_api


signatures
 
?
w_mu
w_sigma
b_mu
b_sigma

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?
w_mu
w_sigma
b_mu
b_sigma

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?
w_mu
w_sigma
b_mu
b_sigma

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
?

#beta_1

$beta_2
	%decay
&learning_rate
'itermAmBmCmDmEmFmGmHmImJmKmLvMvNvOvPvQvRvSvTvUvVvWvX
 
V
0
1
2
3
4
5
6
7
8
9
10
11
V
0
1
2
3
4
5
6
7
8
9
10
11
?
regularization_losses
trainable_variables
(metrics
	variables
)layer_regularization_losses

*layers
+non_trainable_variables
,layer_metrics
 
ZX
VARIABLE_VALUEnoisy_dense/w_mu4layer_with_weights-0/w_mu/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEnoisy_dense/w_sigma7layer_with_weights-0/w_sigma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEnoisy_dense/b_mu4layer_with_weights-0/b_mu/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEnoisy_dense/b_sigma7layer_with_weights-0/b_sigma/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3

0
1
2
3
?
regularization_losses
-metrics
trainable_variables
	variables
.layer_regularization_losses

/layers
0non_trainable_variables
1layer_metrics
\Z
VARIABLE_VALUEnoisy_dense_1/w_mu4layer_with_weights-1/w_mu/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEnoisy_dense_1/w_sigma7layer_with_weights-1/w_sigma/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEnoisy_dense_1/b_mu4layer_with_weights-1/b_mu/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEnoisy_dense_1/b_sigma7layer_with_weights-1/b_sigma/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3

0
1
2
3
?
regularization_losses
2metrics
trainable_variables
	variables
3layer_regularization_losses

4layers
5non_trainable_variables
6layer_metrics
\Z
VARIABLE_VALUEnoisy_dense_2/w_mu4layer_with_weights-2/w_mu/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEnoisy_dense_2/w_sigma7layer_with_weights-2/w_sigma/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEnoisy_dense_2/b_mu4layer_with_weights-2/b_mu/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEnoisy_dense_2/b_sigma7layer_with_weights-2/b_sigma/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3

0
1
2
3
?
regularization_losses
7metrics
 trainable_variables
!	variables
8layer_regularization_losses

9layers
:non_trainable_variables
;layer_metrics
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
<0
 

0
1
2
3
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
	=total
	>count
?	variables
@	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

=0
>1

?	variables
}{
VARIABLE_VALUEAdam/noisy_dense/w_mu/mPlayer_with_weights-0/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/noisy_dense/w_sigma/mSlayer_with_weights-0/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/noisy_dense/b_mu/mPlayer_with_weights-0/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/noisy_dense/b_sigma/mSlayer_with_weights-0/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/noisy_dense_1/w_mu/mPlayer_with_weights-1/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/noisy_dense_1/w_sigma/mSlayer_with_weights-1/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/noisy_dense_1/b_mu/mPlayer_with_weights-1/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/noisy_dense_1/b_sigma/mSlayer_with_weights-1/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/noisy_dense_2/w_mu/mPlayer_with_weights-2/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/noisy_dense_2/w_sigma/mSlayer_with_weights-2/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/noisy_dense_2/b_mu/mPlayer_with_weights-2/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/noisy_dense_2/b_sigma/mSlayer_with_weights-2/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/noisy_dense/w_mu/vPlayer_with_weights-0/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/noisy_dense/w_sigma/vSlayer_with_weights-0/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/noisy_dense/b_mu/vPlayer_with_weights-0/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/noisy_dense/b_sigma/vSlayer_with_weights-0/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/noisy_dense_1/w_mu/vPlayer_with_weights-1/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/noisy_dense_1/w_sigma/vSlayer_with_weights-1/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/noisy_dense_1/b_mu/vPlayer_with_weights-1/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/noisy_dense_1/b_sigma/vSlayer_with_weights-1/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/noisy_dense_2/w_mu/vPlayer_with_weights-2/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/noisy_dense_2/w_sigma/vSlayer_with_weights-2/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/noisy_dense_2/b_mu/vPlayer_with_weights-2/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/noisy_dense_2/b_sigma/vSlayer_with_weights-2/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1noisy_dense/w_munoisy_dense/b_munoisy_dense_1/w_munoisy_dense_1/b_munoisy_dense_2/w_munoisy_dense_2/b_mu*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_3920779
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$noisy_dense/w_mu/Read/ReadVariableOp'noisy_dense/w_sigma/Read/ReadVariableOp$noisy_dense/b_mu/Read/ReadVariableOp'noisy_dense/b_sigma/Read/ReadVariableOp&noisy_dense_1/w_mu/Read/ReadVariableOp)noisy_dense_1/w_sigma/Read/ReadVariableOp&noisy_dense_1/b_mu/Read/ReadVariableOp)noisy_dense_1/b_sigma/Read/ReadVariableOp&noisy_dense_2/w_mu/Read/ReadVariableOp)noisy_dense_2/w_sigma/Read/ReadVariableOp&noisy_dense_2/b_mu/Read/ReadVariableOp)noisy_dense_2/b_sigma/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/noisy_dense/w_mu/m/Read/ReadVariableOp.Adam/noisy_dense/w_sigma/m/Read/ReadVariableOp+Adam/noisy_dense/b_mu/m/Read/ReadVariableOp.Adam/noisy_dense/b_sigma/m/Read/ReadVariableOp-Adam/noisy_dense_1/w_mu/m/Read/ReadVariableOp0Adam/noisy_dense_1/w_sigma/m/Read/ReadVariableOp-Adam/noisy_dense_1/b_mu/m/Read/ReadVariableOp0Adam/noisy_dense_1/b_sigma/m/Read/ReadVariableOp-Adam/noisy_dense_2/w_mu/m/Read/ReadVariableOp0Adam/noisy_dense_2/w_sigma/m/Read/ReadVariableOp-Adam/noisy_dense_2/b_mu/m/Read/ReadVariableOp0Adam/noisy_dense_2/b_sigma/m/Read/ReadVariableOp+Adam/noisy_dense/w_mu/v/Read/ReadVariableOp.Adam/noisy_dense/w_sigma/v/Read/ReadVariableOp+Adam/noisy_dense/b_mu/v/Read/ReadVariableOp.Adam/noisy_dense/b_sigma/v/Read/ReadVariableOp-Adam/noisy_dense_1/w_mu/v/Read/ReadVariableOp0Adam/noisy_dense_1/w_sigma/v/Read/ReadVariableOp-Adam/noisy_dense_1/b_mu/v/Read/ReadVariableOp0Adam/noisy_dense_1/b_sigma/v/Read/ReadVariableOp-Adam/noisy_dense_2/w_mu/v/Read/ReadVariableOp0Adam/noisy_dense_2/w_sigma/v/Read/ReadVariableOp-Adam/noisy_dense_2/b_mu/v/Read/ReadVariableOp0Adam/noisy_dense_2/b_sigma/v/Read/ReadVariableOpConst_6*8
Tin1
/2-	*
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
 __inference__traced_save_3921245
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamenoisy_dense/w_munoisy_dense/w_sigmanoisy_dense/b_munoisy_dense/b_sigmanoisy_dense_1/w_munoisy_dense_1/w_sigmanoisy_dense_1/b_munoisy_dense_1/b_sigmanoisy_dense_2/w_munoisy_dense_2/w_sigmanoisy_dense_2/b_munoisy_dense_2/b_sigmabeta_1beta_2decaylearning_rate	Adam/itertotalcountAdam/noisy_dense/w_mu/mAdam/noisy_dense/w_sigma/mAdam/noisy_dense/b_mu/mAdam/noisy_dense/b_sigma/mAdam/noisy_dense_1/w_mu/mAdam/noisy_dense_1/w_sigma/mAdam/noisy_dense_1/b_mu/mAdam/noisy_dense_1/b_sigma/mAdam/noisy_dense_2/w_mu/mAdam/noisy_dense_2/w_sigma/mAdam/noisy_dense_2/b_mu/mAdam/noisy_dense_2/b_sigma/mAdam/noisy_dense/w_mu/vAdam/noisy_dense/w_sigma/vAdam/noisy_dense/b_mu/vAdam/noisy_dense/b_sigma/vAdam/noisy_dense_1/w_mu/vAdam/noisy_dense_1/w_sigma/vAdam/noisy_dense_1/b_mu/vAdam/noisy_dense_1/b_sigma/vAdam/noisy_dense_2/w_mu/vAdam/noisy_dense_2/w_sigma/vAdam/noisy_dense_2/b_mu/vAdam/noisy_dense_2/b_sigma/v*7
Tin0
.2,*
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
#__inference__traced_restore_3921384??
?
?
J__inference_DQN_Noisy_Net_layer_call_and_return_conditional_losses_3920677

inputs
noisy_dense_3920637
noisy_dense_3920639
noisy_dense_3920641
noisy_dense_3920643
noisy_dense_3920645
noisy_dense_3920647
noisy_dense_1_3920650
noisy_dense_1_3920652
noisy_dense_1_3920654
noisy_dense_1_3920656
noisy_dense_1_3920658
noisy_dense_1_3920660
noisy_dense_2_3920663
noisy_dense_2_3920665
noisy_dense_2_3920667
noisy_dense_2_3920669
noisy_dense_2_3920671
noisy_dense_2_3920673
identity??#noisy_dense/StatefulPartitionedCall?%noisy_dense_1/StatefulPartitionedCall?%noisy_dense_2/StatefulPartitionedCall?
#noisy_dense/StatefulPartitionedCallStatefulPartitionedCallinputsnoisy_dense_3920637noisy_dense_3920639noisy_dense_3920641noisy_dense_3920643noisy_dense_3920645noisy_dense_3920647*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_noisy_dense_layer_call_and_return_conditional_losses_39204152%
#noisy_dense/StatefulPartitionedCall?
%noisy_dense_1/StatefulPartitionedCallStatefulPartitionedCall,noisy_dense/StatefulPartitionedCall:output:0noisy_dense_1_3920650noisy_dense_1_3920652noisy_dense_1_3920654noisy_dense_1_3920656noisy_dense_1_3920658noisy_dense_1_3920660*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_noisy_dense_1_layer_call_and_return_conditional_losses_39204882'
%noisy_dense_1/StatefulPartitionedCall?
%noisy_dense_2/StatefulPartitionedCallStatefulPartitionedCall.noisy_dense_1/StatefulPartitionedCall:output:0noisy_dense_2_3920663noisy_dense_2_3920665noisy_dense_2_3920667noisy_dense_2_3920669noisy_dense_2_3920671noisy_dense_2_3920673*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_noisy_dense_2_layer_call_and_return_conditional_losses_39205602'
%noisy_dense_2/StatefulPartitionedCall?
IdentityIdentity.noisy_dense_2/StatefulPartitionedCall:output:0$^noisy_dense/StatefulPartitionedCall&^noisy_dense_1/StatefulPartitionedCall&^noisy_dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????:::::::: ::: ::: ::::2J
#noisy_dense/StatefulPartitionedCall#noisy_dense/StatefulPartitionedCall2N
%noisy_dense_1/StatefulPartitionedCall%noisy_dense_1/StatefulPartitionedCall2N
%noisy_dense_2/StatefulPartitionedCall%noisy_dense_2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
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

: : 

_output_shapes
:
?@
?	
J__inference_DQN_Noisy_Net_layer_call_and_return_conditional_losses_3920833

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
+noisy_dense_2_add_1_readvariableop_resource
identity??noisy_dense/Add/ReadVariableOp? noisy_dense/Add_1/ReadVariableOp?noisy_dense/Mul/ReadVariableOp? noisy_dense/Mul_1/ReadVariableOp? noisy_dense_1/Add/ReadVariableOp?"noisy_dense_1/Add_1/ReadVariableOp? noisy_dense_1/Mul/ReadVariableOp?"noisy_dense_1/Mul_1/ReadVariableOp? noisy_dense_2/Add/ReadVariableOp?"noisy_dense_2/Add_1/ReadVariableOp? noisy_dense_2/Mul/ReadVariableOp?"noisy_dense_2/Mul_1/ReadVariableOp?
noisy_dense/Mul/ReadVariableOpReadVariableOp'noisy_dense_mul_readvariableop_resource*
_output_shapes

:*
dtype02 
noisy_dense/Mul/ReadVariableOp?
noisy_dense/MulMul&noisy_dense/Mul/ReadVariableOp:value:0noisy_dense_mul_y*
T0*
_output_shapes

:2
noisy_dense/Mul?
noisy_dense/Add/ReadVariableOpReadVariableOp'noisy_dense_add_readvariableop_resource*
_output_shapes

:*
dtype02 
noisy_dense/Add/ReadVariableOp?
noisy_dense/AddAdd&noisy_dense/Add/ReadVariableOp:value:0noisy_dense/Mul:z:0*
T0*
_output_shapes

:2
noisy_dense/Add?
noisy_dense/MatMulMatMulinputsnoisy_dense/Add:z:0*
T0*'
_output_shapes
:?????????2
noisy_dense/MatMul?
 noisy_dense/Mul_1/ReadVariableOpReadVariableOp)noisy_dense_mul_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 noisy_dense/Mul_1/ReadVariableOp?
noisy_dense/Mul_1Mul(noisy_dense/Mul_1/ReadVariableOp:value:0noisy_dense_mul_1_y*
T0*
_output_shapes
:2
noisy_dense/Mul_1?
 noisy_dense/Add_1/ReadVariableOpReadVariableOp)noisy_dense_add_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 noisy_dense/Add_1/ReadVariableOp?
noisy_dense/Add_1Add(noisy_dense/Add_1/ReadVariableOp:value:0noisy_dense/Mul_1:z:0*
T0*
_output_shapes
:2
noisy_dense/Add_1?
noisy_dense/BiasAddBiasAddnoisy_dense/MatMul:product:0noisy_dense/Add_1:z:0*
T0*'
_output_shapes
:?????????2
noisy_dense/BiasAdd|
noisy_dense/ReluRelunoisy_dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
noisy_dense/Relu?
 noisy_dense_1/Mul/ReadVariableOpReadVariableOp)noisy_dense_1_mul_readvariableop_resource*
_output_shapes

: *
dtype02"
 noisy_dense_1/Mul/ReadVariableOp?
noisy_dense_1/MulMul(noisy_dense_1/Mul/ReadVariableOp:value:0noisy_dense_1_mul_y*
T0*
_output_shapes

: 2
noisy_dense_1/Mul?
 noisy_dense_1/Add/ReadVariableOpReadVariableOp)noisy_dense_1_add_readvariableop_resource*
_output_shapes

: *
dtype02"
 noisy_dense_1/Add/ReadVariableOp?
noisy_dense_1/AddAdd(noisy_dense_1/Add/ReadVariableOp:value:0noisy_dense_1/Mul:z:0*
T0*
_output_shapes

: 2
noisy_dense_1/Add?
noisy_dense_1/MatMulMatMulnoisy_dense/Relu:activations:0noisy_dense_1/Add:z:0*
T0*'
_output_shapes
:????????? 2
noisy_dense_1/MatMul?
"noisy_dense_1/Mul_1/ReadVariableOpReadVariableOp+noisy_dense_1_mul_1_readvariableop_resource*
_output_shapes
: *
dtype02$
"noisy_dense_1/Mul_1/ReadVariableOp?
noisy_dense_1/Mul_1Mul*noisy_dense_1/Mul_1/ReadVariableOp:value:0noisy_dense_1_mul_1_y*
T0*
_output_shapes
: 2
noisy_dense_1/Mul_1?
"noisy_dense_1/Add_1/ReadVariableOpReadVariableOp+noisy_dense_1_add_1_readvariableop_resource*
_output_shapes
: *
dtype02$
"noisy_dense_1/Add_1/ReadVariableOp?
noisy_dense_1/Add_1Add*noisy_dense_1/Add_1/ReadVariableOp:value:0noisy_dense_1/Mul_1:z:0*
T0*
_output_shapes
: 2
noisy_dense_1/Add_1?
noisy_dense_1/BiasAddBiasAddnoisy_dense_1/MatMul:product:0noisy_dense_1/Add_1:z:0*
T0*'
_output_shapes
:????????? 2
noisy_dense_1/BiasAdd?
noisy_dense_1/ReluRelunoisy_dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
noisy_dense_1/Relu?
 noisy_dense_2/Mul/ReadVariableOpReadVariableOp)noisy_dense_2_mul_readvariableop_resource*
_output_shapes

: *
dtype02"
 noisy_dense_2/Mul/ReadVariableOp?
noisy_dense_2/MulMul(noisy_dense_2/Mul/ReadVariableOp:value:0noisy_dense_2_mul_y*
T0*
_output_shapes

: 2
noisy_dense_2/Mul?
 noisy_dense_2/Add/ReadVariableOpReadVariableOp)noisy_dense_2_add_readvariableop_resource*
_output_shapes

: *
dtype02"
 noisy_dense_2/Add/ReadVariableOp?
noisy_dense_2/AddAdd(noisy_dense_2/Add/ReadVariableOp:value:0noisy_dense_2/Mul:z:0*
T0*
_output_shapes

: 2
noisy_dense_2/Add?
noisy_dense_2/MatMulMatMul noisy_dense_1/Relu:activations:0noisy_dense_2/Add:z:0*
T0*'
_output_shapes
:?????????2
noisy_dense_2/MatMul?
"noisy_dense_2/Mul_1/ReadVariableOpReadVariableOp+noisy_dense_2_mul_1_readvariableop_resource*
_output_shapes
:*
dtype02$
"noisy_dense_2/Mul_1/ReadVariableOp?
noisy_dense_2/Mul_1Mul*noisy_dense_2/Mul_1/ReadVariableOp:value:0noisy_dense_2_mul_1_y*
T0*
_output_shapes
:2
noisy_dense_2/Mul_1?
"noisy_dense_2/Add_1/ReadVariableOpReadVariableOp+noisy_dense_2_add_1_readvariableop_resource*
_output_shapes
:*
dtype02$
"noisy_dense_2/Add_1/ReadVariableOp?
noisy_dense_2/Add_1Add*noisy_dense_2/Add_1/ReadVariableOp:value:0noisy_dense_2/Mul_1:z:0*
T0*
_output_shapes
:2
noisy_dense_2/Add_1?
noisy_dense_2/BiasAddBiasAddnoisy_dense_2/MatMul:product:0noisy_dense_2/Add_1:z:0*
T0*'
_output_shapes
:?????????2
noisy_dense_2/BiasAdd?
IdentityIdentitynoisy_dense_2/BiasAdd:output:0^noisy_dense/Add/ReadVariableOp!^noisy_dense/Add_1/ReadVariableOp^noisy_dense/Mul/ReadVariableOp!^noisy_dense/Mul_1/ReadVariableOp!^noisy_dense_1/Add/ReadVariableOp#^noisy_dense_1/Add_1/ReadVariableOp!^noisy_dense_1/Mul/ReadVariableOp#^noisy_dense_1/Mul_1/ReadVariableOp!^noisy_dense_2/Add/ReadVariableOp#^noisy_dense_2/Add_1/ReadVariableOp!^noisy_dense_2/Mul/ReadVariableOp#^noisy_dense_2/Mul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????:::::::: ::: ::: ::::2@
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
"noisy_dense_2/Mul_1/ReadVariableOp"noisy_dense_2/Mul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
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

: : 

_output_shapes
:
?	
?
H__inference_noisy_dense_layer_call_and_return_conditional_losses_3920426

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
?
?
J__inference_noisy_dense_2_layer_call_and_return_conditional_losses_3921051

inputs
mul_readvariableop_resource	
mul_y
add_readvariableop_resource!
mul_1_readvariableop_resource
mul_1_y!
add_1_readvariableop_resource
identity??Add/ReadVariableOp?Add_1/ReadVariableOp?Mul/ReadVariableOp?Mul_1/ReadVariableOp?
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

: *
dtype02
Mul/ReadVariableOp]
MulMulMul/ReadVariableOp:value:0mul_y*
T0*
_output_shapes

: 2
Mul?
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

: *
dtype02
Add/ReadVariableOp_
AddAddAdd/ReadVariableOp:value:0Mul:z:0*
T0*
_output_shapes

: 2
Add]
MatMulMatMulinputsAdd:z:0*
T0*'
_output_shapes
:?????????2
MatMul?
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
:*
dtype02
Mul_1/ReadVariableOpa
Mul_1MulMul_1/ReadVariableOp:value:0mul_1_y*
T0*
_output_shapes
:2
Mul_1?
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype02
Add_1/ReadVariableOpc
Add_1AddAdd_1/ReadVariableOp:value:0	Mul_1:z:0*
T0*
_output_shapes
:2
Add_1l
BiasAddBiasAddMatMul:product:0	Add_1:z:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Mul/ReadVariableOp^Mul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:????????? :: ::::2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:$ 

_output_shapes

: : 

_output_shapes
:
?
?
H__inference_noisy_dense_layer_call_and_return_conditional_losses_3920936

inputs
mul_readvariableop_resource	
mul_y
add_readvariableop_resource!
mul_1_readvariableop_resource
mul_1_y!
add_1_readvariableop_resource
identity??Add/ReadVariableOp?Add_1/ReadVariableOp?Mul/ReadVariableOp?Mul_1/ReadVariableOp?
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

:*
dtype02
Mul/ReadVariableOp]
MulMulMul/ReadVariableOp:value:0mul_y*
T0*
_output_shapes

:2
Mul?
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
:?????????2
MatMul?
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
:*
dtype02
Mul_1/ReadVariableOpa
Mul_1MulMul_1/ReadVariableOp:value:0mul_1_y*
T0*
_output_shapes
:2
Mul_1?
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
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Mul/ReadVariableOp^Mul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:$ 

_output_shapes

:: 

_output_shapes
:
?
?
J__inference_noisy_dense_1_layer_call_and_return_conditional_losses_3920488

inputs
mul_readvariableop_resource	
mul_y
add_readvariableop_resource!
mul_1_readvariableop_resource
mul_1_y!
add_1_readvariableop_resource
identity??Add/ReadVariableOp?Add_1/ReadVariableOp?Mul/ReadVariableOp?Mul_1/ReadVariableOp?
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

: *
dtype02
Mul/ReadVariableOp]
MulMulMul/ReadVariableOp:value:0mul_y*
T0*
_output_shapes

: 2
Mul?
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
:????????? 2
MatMul?
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
: *
dtype02
Mul_1/ReadVariableOpa
Mul_1MulMul_1/ReadVariableOp:value:0mul_1_y*
T0*
_output_shapes
: 2
Mul_1?
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
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Mul/ReadVariableOp^Mul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????:: ::: :2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:$ 

_output_shapes

: : 

_output_shapes
: 
?
?
J__inference_DQN_Noisy_Net_layer_call_and_return_conditional_losses_3920631
input_1
noisy_dense_3920615
noisy_dense_3920617
noisy_dense_1_3920620
noisy_dense_1_3920622
noisy_dense_2_3920625
noisy_dense_2_3920627
identity??#noisy_dense/StatefulPartitionedCall?%noisy_dense_1/StatefulPartitionedCall?%noisy_dense_2/StatefulPartitionedCall?
#noisy_dense/StatefulPartitionedCallStatefulPartitionedCallinput_1noisy_dense_3920615noisy_dense_3920617*
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
GPU 2J 8? *Q
fLRJ
H__inference_noisy_dense_layer_call_and_return_conditional_losses_39204262%
#noisy_dense/StatefulPartitionedCall?
%noisy_dense_1/StatefulPartitionedCallStatefulPartitionedCall,noisy_dense/StatefulPartitionedCall:output:0noisy_dense_1_3920620noisy_dense_1_3920622*
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
GPU 2J 8? *S
fNRL
J__inference_noisy_dense_1_layer_call_and_return_conditional_losses_39204992'
%noisy_dense_1/StatefulPartitionedCall?
%noisy_dense_2/StatefulPartitionedCallStatefulPartitionedCall.noisy_dense_1/StatefulPartitionedCall:output:0noisy_dense_2_3920625noisy_dense_2_3920627*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_noisy_dense_2_layer_call_and_return_conditional_losses_39205702'
%noisy_dense_2/StatefulPartitionedCall?
IdentityIdentity.noisy_dense_2/StatefulPartitionedCall:output:0$^noisy_dense/StatefulPartitionedCall&^noisy_dense_1/StatefulPartitionedCall&^noisy_dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2J
#noisy_dense/StatefulPartitionedCall#noisy_dense/StatefulPartitionedCall2N
%noisy_dense_1/StatefulPartitionedCall%noisy_dense_1/StatefulPartitionedCall2N
%noisy_dense_2/StatefulPartitionedCall%noisy_dense_2/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
J__inference_noisy_dense_2_layer_call_and_return_conditional_losses_3920560

inputs
mul_readvariableop_resource	
mul_y
add_readvariableop_resource!
mul_1_readvariableop_resource
mul_1_y!
add_1_readvariableop_resource
identity??Add/ReadVariableOp?Add_1/ReadVariableOp?Mul/ReadVariableOp?Mul_1/ReadVariableOp?
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

: *
dtype02
Mul/ReadVariableOp]
MulMulMul/ReadVariableOp:value:0mul_y*
T0*
_output_shapes

: 2
Mul?
Add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

: *
dtype02
Add/ReadVariableOp_
AddAddAdd/ReadVariableOp:value:0Mul:z:0*
T0*
_output_shapes

: 2
Add]
MatMulMatMulinputsAdd:z:0*
T0*'
_output_shapes
:?????????2
MatMul?
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
:*
dtype02
Mul_1/ReadVariableOpa
Mul_1MulMul_1/ReadVariableOp:value:0mul_1_y*
T0*
_output_shapes
:2
Mul_1?
Add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype02
Add_1/ReadVariableOpc
Add_1AddAdd_1/ReadVariableOp:value:0	Mul_1:z:0*
T0*
_output_shapes
:2
Add_1l
BiasAddBiasAddMatMul:product:0	Add_1:z:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Mul/ReadVariableOp^Mul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:????????? :: ::::2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:$ 

_output_shapes

: : 

_output_shapes
:
?
?
/__inference_DQN_Noisy_Net_layer_call_fn_3920898

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

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_DQN_Noisy_Net_layer_call_and_return_conditional_losses_39206772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????:::::::: ::: ::: ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
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

: : 

_output_shapes
:
?
?
/__inference_DQN_Noisy_Net_layer_call_fn_3920716
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

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_DQN_Noisy_Net_layer_call_and_return_conditional_losses_39206772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????:::::::: ::: ::: ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
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

: : 

_output_shapes
:
?
?
/__inference_DQN_Noisy_Net_layer_call_fn_3920915

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_DQN_Noisy_Net_layer_call_and_return_conditional_losses_39207372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_DQN_Noisy_Net_layer_call_and_return_conditional_losses_3920612
input_1
noisy_dense_3920454
noisy_dense_3920456
noisy_dense_3920458
noisy_dense_3920460
noisy_dense_3920462
noisy_dense_3920464
noisy_dense_1_3920527
noisy_dense_1_3920529
noisy_dense_1_3920531
noisy_dense_1_3920533
noisy_dense_1_3920535
noisy_dense_1_3920537
noisy_dense_2_3920598
noisy_dense_2_3920600
noisy_dense_2_3920602
noisy_dense_2_3920604
noisy_dense_2_3920606
noisy_dense_2_3920608
identity??#noisy_dense/StatefulPartitionedCall?%noisy_dense_1/StatefulPartitionedCall?%noisy_dense_2/StatefulPartitionedCall?
#noisy_dense/StatefulPartitionedCallStatefulPartitionedCallinput_1noisy_dense_3920454noisy_dense_3920456noisy_dense_3920458noisy_dense_3920460noisy_dense_3920462noisy_dense_3920464*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_noisy_dense_layer_call_and_return_conditional_losses_39204152%
#noisy_dense/StatefulPartitionedCall?
%noisy_dense_1/StatefulPartitionedCallStatefulPartitionedCall,noisy_dense/StatefulPartitionedCall:output:0noisy_dense_1_3920527noisy_dense_1_3920529noisy_dense_1_3920531noisy_dense_1_3920533noisy_dense_1_3920535noisy_dense_1_3920537*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_noisy_dense_1_layer_call_and_return_conditional_losses_39204882'
%noisy_dense_1/StatefulPartitionedCall?
%noisy_dense_2/StatefulPartitionedCallStatefulPartitionedCall.noisy_dense_1/StatefulPartitionedCall:output:0noisy_dense_2_3920598noisy_dense_2_3920600noisy_dense_2_3920602noisy_dense_2_3920604noisy_dense_2_3920606noisy_dense_2_3920608*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_noisy_dense_2_layer_call_and_return_conditional_losses_39205602'
%noisy_dense_2/StatefulPartitionedCall?
IdentityIdentity.noisy_dense_2/StatefulPartitionedCall:output:0$^noisy_dense/StatefulPartitionedCall&^noisy_dense_1/StatefulPartitionedCall&^noisy_dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????:::::::: ::: ::: ::::2J
#noisy_dense/StatefulPartitionedCall#noisy_dense/StatefulPartitionedCall2N
%noisy_dense_1/StatefulPartitionedCall%noisy_dense_1/StatefulPartitionedCall2N
%noisy_dense_2/StatefulPartitionedCall%noisy_dense_2/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
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

: : 

_output_shapes
:
?
?
-__inference_noisy_dense_layer_call_fn_3920973

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
GPU 2J 8? *Q
fLRJ
H__inference_noisy_dense_layer_call_and_return_conditional_losses_39204262
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
?
?
/__inference_noisy_dense_1_layer_call_fn_3921031

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
GPU 2J 8? *S
fNRL
J__inference_noisy_dense_1_layer_call_and_return_conditional_losses_39204992
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
?
?
/__inference_noisy_dense_2_layer_call_fn_3921078

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_noisy_dense_2_layer_call_and_return_conditional_losses_39205602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:????????? :: ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:$ 

_output_shapes

: : 

_output_shapes
:
?	
?
J__inference_noisy_dense_2_layer_call_and_return_conditional_losses_3920570

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
?
?
H__inference_noisy_dense_layer_call_and_return_conditional_losses_3920415

inputs
mul_readvariableop_resource	
mul_y
add_readvariableop_resource!
mul_1_readvariableop_resource
mul_1_y!
add_1_readvariableop_resource
identity??Add/ReadVariableOp?Add_1/ReadVariableOp?Mul/ReadVariableOp?Mul_1/ReadVariableOp?
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

:*
dtype02
Mul/ReadVariableOp]
MulMulMul/ReadVariableOp:value:0mul_y*
T0*
_output_shapes

:2
Mul?
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
:?????????2
MatMul?
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
:*
dtype02
Mul_1/ReadVariableOpa
Mul_1MulMul_1/ReadVariableOp:value:0mul_1_y*
T0*
_output_shapes
:2
Mul_1?
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
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Mul/ReadVariableOp^Mul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:$ 

_output_shapes

:: 

_output_shapes
:
?
?
J__inference_DQN_Noisy_Net_layer_call_and_return_conditional_losses_3920737

inputs
noisy_dense_3920721
noisy_dense_3920723
noisy_dense_1_3920726
noisy_dense_1_3920728
noisy_dense_2_3920731
noisy_dense_2_3920733
identity??#noisy_dense/StatefulPartitionedCall?%noisy_dense_1/StatefulPartitionedCall?%noisy_dense_2/StatefulPartitionedCall?
#noisy_dense/StatefulPartitionedCallStatefulPartitionedCallinputsnoisy_dense_3920721noisy_dense_3920723*
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
GPU 2J 8? *Q
fLRJ
H__inference_noisy_dense_layer_call_and_return_conditional_losses_39204262%
#noisy_dense/StatefulPartitionedCall?
%noisy_dense_1/StatefulPartitionedCallStatefulPartitionedCall,noisy_dense/StatefulPartitionedCall:output:0noisy_dense_1_3920726noisy_dense_1_3920728*
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
GPU 2J 8? *S
fNRL
J__inference_noisy_dense_1_layer_call_and_return_conditional_losses_39204992'
%noisy_dense_1/StatefulPartitionedCall?
%noisy_dense_2/StatefulPartitionedCallStatefulPartitionedCall.noisy_dense_1/StatefulPartitionedCall:output:0noisy_dense_2_3920731noisy_dense_2_3920733*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_noisy_dense_2_layer_call_and_return_conditional_losses_39205702'
%noisy_dense_2/StatefulPartitionedCall?
IdentityIdentity.noisy_dense_2/StatefulPartitionedCall:output:0$^noisy_dense/StatefulPartitionedCall&^noisy_dense_1/StatefulPartitionedCall&^noisy_dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2J
#noisy_dense/StatefulPartitionedCall#noisy_dense/StatefulPartitionedCall2N
%noisy_dense_1/StatefulPartitionedCall%noisy_dense_1/StatefulPartitionedCall2N
%noisy_dense_2/StatefulPartitionedCall%noisy_dense_2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_noisy_dense_layer_call_fn_3920964

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_noisy_dense_layer_call_and_return_conditional_losses_39204152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:$ 

_output_shapes

:: 

_output_shapes
:
?	
?
J__inference_noisy_dense_1_layer_call_and_return_conditional_losses_3921005

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
?
?
/__inference_noisy_dense_2_layer_call_fn_3921087

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
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_noisy_dense_2_layer_call_and_return_conditional_losses_39205702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
J__inference_noisy_dense_2_layer_call_and_return_conditional_losses_3921061

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
??
?
#__inference__traced_restore_3921384
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
)assignvariableop_11_noisy_dense_2_b_sigma
assignvariableop_12_beta_1
assignvariableop_13_beta_2
assignvariableop_14_decay%
!assignvariableop_15_learning_rate!
assignvariableop_16_adam_iter
assignvariableop_17_total
assignvariableop_18_count/
+assignvariableop_19_adam_noisy_dense_w_mu_m2
.assignvariableop_20_adam_noisy_dense_w_sigma_m/
+assignvariableop_21_adam_noisy_dense_b_mu_m2
.assignvariableop_22_adam_noisy_dense_b_sigma_m1
-assignvariableop_23_adam_noisy_dense_1_w_mu_m4
0assignvariableop_24_adam_noisy_dense_1_w_sigma_m1
-assignvariableop_25_adam_noisy_dense_1_b_mu_m4
0assignvariableop_26_adam_noisy_dense_1_b_sigma_m1
-assignvariableop_27_adam_noisy_dense_2_w_mu_m4
0assignvariableop_28_adam_noisy_dense_2_w_sigma_m1
-assignvariableop_29_adam_noisy_dense_2_b_mu_m4
0assignvariableop_30_adam_noisy_dense_2_b_sigma_m/
+assignvariableop_31_adam_noisy_dense_w_mu_v2
.assignvariableop_32_adam_noisy_dense_w_sigma_v/
+assignvariableop_33_adam_noisy_dense_b_mu_v2
.assignvariableop_34_adam_noisy_dense_b_sigma_v1
-assignvariableop_35_adam_noisy_dense_1_w_mu_v4
0assignvariableop_36_adam_noisy_dense_1_w_sigma_v1
-assignvariableop_37_adam_noisy_dense_1_b_mu_v4
0assignvariableop_38_adam_noisy_dense_1_b_sigma_v1
-assignvariableop_39_adam_noisy_dense_2_w_mu_v4
0assignvariableop_40_adam_noisy_dense_2_w_sigma_v1
-assignvariableop_41_adam_noisy_dense_2_b_mu_v4
0assignvariableop_42_adam_noisy_dense_2_b_sigma_v
identity_44??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*?
value?B?,B4layer_with_weights-0/w_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-0/w_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/b_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-0/b_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/w_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-1/w_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/b_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-1/b_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/w_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-2/w_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/b_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-2/b_sigma/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_noisy_dense_w_muIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp&assignvariableop_1_noisy_dense_w_sigmaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_noisy_dense_b_muIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp&assignvariableop_3_noisy_dense_b_sigmaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_noisy_dense_1_w_muIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp(assignvariableop_5_noisy_dense_1_w_sigmaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_noisy_dense_1_b_muIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp(assignvariableop_7_noisy_dense_1_b_sigmaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_noisy_dense_2_w_muIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp(assignvariableop_9_noisy_dense_2_w_sigmaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_noisy_dense_2_b_muIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp)assignvariableop_11_noisy_dense_2_b_sigmaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_beta_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_beta_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_decayIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_learning_rateIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_noisy_dense_w_mu_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp.assignvariableop_20_adam_noisy_dense_w_sigma_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_noisy_dense_b_mu_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp.assignvariableop_22_adam_noisy_dense_b_sigma_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp-assignvariableop_23_adam_noisy_dense_1_w_mu_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp0assignvariableop_24_adam_noisy_dense_1_w_sigma_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp-assignvariableop_25_adam_noisy_dense_1_b_mu_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp0assignvariableop_26_adam_noisy_dense_1_b_sigma_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp-assignvariableop_27_adam_noisy_dense_2_w_mu_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp0assignvariableop_28_adam_noisy_dense_2_w_sigma_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp-assignvariableop_29_adam_noisy_dense_2_b_mu_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp0assignvariableop_30_adam_noisy_dense_2_b_sigma_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_noisy_dense_w_mu_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp.assignvariableop_32_adam_noisy_dense_w_sigma_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_noisy_dense_b_mu_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp.assignvariableop_34_adam_noisy_dense_b_sigma_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp-assignvariableop_35_adam_noisy_dense_1_w_mu_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp0assignvariableop_36_adam_noisy_dense_1_w_sigma_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp-assignvariableop_37_adam_noisy_dense_1_b_mu_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp0assignvariableop_38_adam_noisy_dense_1_b_sigma_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp-assignvariableop_39_adam_noisy_dense_2_w_mu_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp0assignvariableop_40_adam_noisy_dense_2_w_sigma_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp-assignvariableop_41_adam_noisy_dense_2_b_mu_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp0assignvariableop_42_adam_noisy_dense_2_b_sigma_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_429
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_43?
Identity_44IdentityIdentity_43:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_44"#
identity_44Identity_44:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_42AssignVariableOp_422(
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
?'
?
"__inference__wrapped_model_3920390
input_1<
8dqn_noisy_net_noisy_dense_matmul_readvariableop_resource=
9dqn_noisy_net_noisy_dense_biasadd_readvariableop_resource>
:dqn_noisy_net_noisy_dense_1_matmul_readvariableop_resource?
;dqn_noisy_net_noisy_dense_1_biasadd_readvariableop_resource>
:dqn_noisy_net_noisy_dense_2_matmul_readvariableop_resource?
;dqn_noisy_net_noisy_dense_2_biasadd_readvariableop_resource
identity??0DQN_Noisy_Net/noisy_dense/BiasAdd/ReadVariableOp?/DQN_Noisy_Net/noisy_dense/MatMul/ReadVariableOp?2DQN_Noisy_Net/noisy_dense_1/BiasAdd/ReadVariableOp?1DQN_Noisy_Net/noisy_dense_1/MatMul/ReadVariableOp?2DQN_Noisy_Net/noisy_dense_2/BiasAdd/ReadVariableOp?1DQN_Noisy_Net/noisy_dense_2/MatMul/ReadVariableOp?
/DQN_Noisy_Net/noisy_dense/MatMul/ReadVariableOpReadVariableOp8dqn_noisy_net_noisy_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype021
/DQN_Noisy_Net/noisy_dense/MatMul/ReadVariableOp?
 DQN_Noisy_Net/noisy_dense/MatMulMatMulinput_17DQN_Noisy_Net/noisy_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 DQN_Noisy_Net/noisy_dense/MatMul?
0DQN_Noisy_Net/noisy_dense/BiasAdd/ReadVariableOpReadVariableOp9dqn_noisy_net_noisy_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0DQN_Noisy_Net/noisy_dense/BiasAdd/ReadVariableOp?
!DQN_Noisy_Net/noisy_dense/BiasAddBiasAdd*DQN_Noisy_Net/noisy_dense/MatMul:product:08DQN_Noisy_Net/noisy_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!DQN_Noisy_Net/noisy_dense/BiasAdd?
DQN_Noisy_Net/noisy_dense/ReluRelu*DQN_Noisy_Net/noisy_dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2 
DQN_Noisy_Net/noisy_dense/Relu?
1DQN_Noisy_Net/noisy_dense_1/MatMul/ReadVariableOpReadVariableOp:dqn_noisy_net_noisy_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype023
1DQN_Noisy_Net/noisy_dense_1/MatMul/ReadVariableOp?
"DQN_Noisy_Net/noisy_dense_1/MatMulMatMul,DQN_Noisy_Net/noisy_dense/Relu:activations:09DQN_Noisy_Net/noisy_dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2$
"DQN_Noisy_Net/noisy_dense_1/MatMul?
2DQN_Noisy_Net/noisy_dense_1/BiasAdd/ReadVariableOpReadVariableOp;dqn_noisy_net_noisy_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype024
2DQN_Noisy_Net/noisy_dense_1/BiasAdd/ReadVariableOp?
#DQN_Noisy_Net/noisy_dense_1/BiasAddBiasAdd,DQN_Noisy_Net/noisy_dense_1/MatMul:product:0:DQN_Noisy_Net/noisy_dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2%
#DQN_Noisy_Net/noisy_dense_1/BiasAdd?
 DQN_Noisy_Net/noisy_dense_1/ReluRelu,DQN_Noisy_Net/noisy_dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2"
 DQN_Noisy_Net/noisy_dense_1/Relu?
1DQN_Noisy_Net/noisy_dense_2/MatMul/ReadVariableOpReadVariableOp:dqn_noisy_net_noisy_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype023
1DQN_Noisy_Net/noisy_dense_2/MatMul/ReadVariableOp?
"DQN_Noisy_Net/noisy_dense_2/MatMulMatMul.DQN_Noisy_Net/noisy_dense_1/Relu:activations:09DQN_Noisy_Net/noisy_dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"DQN_Noisy_Net/noisy_dense_2/MatMul?
2DQN_Noisy_Net/noisy_dense_2/BiasAdd/ReadVariableOpReadVariableOp;dqn_noisy_net_noisy_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2DQN_Noisy_Net/noisy_dense_2/BiasAdd/ReadVariableOp?
#DQN_Noisy_Net/noisy_dense_2/BiasAddBiasAdd,DQN_Noisy_Net/noisy_dense_2/MatMul:product:0:DQN_Noisy_Net/noisy_dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#DQN_Noisy_Net/noisy_dense_2/BiasAdd?
IdentityIdentity,DQN_Noisy_Net/noisy_dense_2/BiasAdd:output:01^DQN_Noisy_Net/noisy_dense/BiasAdd/ReadVariableOp0^DQN_Noisy_Net/noisy_dense/MatMul/ReadVariableOp3^DQN_Noisy_Net/noisy_dense_1/BiasAdd/ReadVariableOp2^DQN_Noisy_Net/noisy_dense_1/MatMul/ReadVariableOp3^DQN_Noisy_Net/noisy_dense_2/BiasAdd/ReadVariableOp2^DQN_Noisy_Net/noisy_dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2d
0DQN_Noisy_Net/noisy_dense/BiasAdd/ReadVariableOp0DQN_Noisy_Net/noisy_dense/BiasAdd/ReadVariableOp2b
/DQN_Noisy_Net/noisy_dense/MatMul/ReadVariableOp/DQN_Noisy_Net/noisy_dense/MatMul/ReadVariableOp2h
2DQN_Noisy_Net/noisy_dense_1/BiasAdd/ReadVariableOp2DQN_Noisy_Net/noisy_dense_1/BiasAdd/ReadVariableOp2f
1DQN_Noisy_Net/noisy_dense_1/MatMul/ReadVariableOp1DQN_Noisy_Net/noisy_dense_1/MatMul/ReadVariableOp2h
2DQN_Noisy_Net/noisy_dense_2/BiasAdd/ReadVariableOp2DQN_Noisy_Net/noisy_dense_2/BiasAdd/ReadVariableOp2f
1DQN_Noisy_Net/noisy_dense_2/MatMul/ReadVariableOp1DQN_Noisy_Net/noisy_dense_2/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
J__inference_noisy_dense_1_layer_call_and_return_conditional_losses_3920499

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
?
?
%__inference_signature_wrapper_3920779
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_39203902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
/__inference_noisy_dense_1_layer_call_fn_3921022

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_noisy_dense_1_layer_call_and_return_conditional_losses_39204882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????:: ::: :22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:$ 

_output_shapes

: : 

_output_shapes
: 
?
?
J__inference_noisy_dense_1_layer_call_and_return_conditional_losses_3920994

inputs
mul_readvariableop_resource	
mul_y
add_readvariableop_resource!
mul_1_readvariableop_resource
mul_1_y!
add_1_readvariableop_resource
identity??Add/ReadVariableOp?Add_1/ReadVariableOp?Mul/ReadVariableOp?Mul_1/ReadVariableOp?
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

: *
dtype02
Mul/ReadVariableOp]
MulMulMul/ReadVariableOp:value:0mul_y*
T0*
_output_shapes

: 2
Mul?
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
:????????? 2
MatMul?
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
: *
dtype02
Mul_1/ReadVariableOpa
Mul_1MulMul_1/ReadVariableOp:value:0mul_1_y*
T0*
_output_shapes
: 2
Mul_1?
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
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^Add/ReadVariableOp^Add_1/ReadVariableOp^Mul/ReadVariableOp^Mul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????:: ::: :2(
Add/ReadVariableOpAdd/ReadVariableOp2,
Add_1/ReadVariableOpAdd_1/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:$ 

_output_shapes

: : 

_output_shapes
: 
?	
?
H__inference_noisy_dense_layer_call_and_return_conditional_losses_3920947

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
?\
?
 __inference__traced_save_3921245
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
0savev2_noisy_dense_2_b_sigma_read_readvariableop%
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
7savev2_adam_noisy_dense_2_b_sigma_m_read_readvariableop6
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
7savev2_adam_noisy_dense_2_b_sigma_v_read_readvariableop
savev2_const_6

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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*?
value?B?,B4layer_with_weights-0/w_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-0/w_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/b_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-0/b_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/w_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-1/w_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/b_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-1/b_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/w_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-2/w_sigma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/b_mu/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-2/b_sigma/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/w_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/w_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/b_mu/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/b_sigma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/w_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/w_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/b_mu/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/b_sigma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_noisy_dense_w_mu_read_readvariableop.savev2_noisy_dense_w_sigma_read_readvariableop+savev2_noisy_dense_b_mu_read_readvariableop.savev2_noisy_dense_b_sigma_read_readvariableop-savev2_noisy_dense_1_w_mu_read_readvariableop0savev2_noisy_dense_1_w_sigma_read_readvariableop-savev2_noisy_dense_1_b_mu_read_readvariableop0savev2_noisy_dense_1_b_sigma_read_readvariableop-savev2_noisy_dense_2_w_mu_read_readvariableop0savev2_noisy_dense_2_w_sigma_read_readvariableop-savev2_noisy_dense_2_b_mu_read_readvariableop0savev2_noisy_dense_2_b_sigma_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_noisy_dense_w_mu_m_read_readvariableop5savev2_adam_noisy_dense_w_sigma_m_read_readvariableop2savev2_adam_noisy_dense_b_mu_m_read_readvariableop5savev2_adam_noisy_dense_b_sigma_m_read_readvariableop4savev2_adam_noisy_dense_1_w_mu_m_read_readvariableop7savev2_adam_noisy_dense_1_w_sigma_m_read_readvariableop4savev2_adam_noisy_dense_1_b_mu_m_read_readvariableop7savev2_adam_noisy_dense_1_b_sigma_m_read_readvariableop4savev2_adam_noisy_dense_2_w_mu_m_read_readvariableop7savev2_adam_noisy_dense_2_w_sigma_m_read_readvariableop4savev2_adam_noisy_dense_2_b_mu_m_read_readvariableop7savev2_adam_noisy_dense_2_b_sigma_m_read_readvariableop2savev2_adam_noisy_dense_w_mu_v_read_readvariableop5savev2_adam_noisy_dense_w_sigma_v_read_readvariableop2savev2_adam_noisy_dense_b_mu_v_read_readvariableop5savev2_adam_noisy_dense_b_sigma_v_read_readvariableop4savev2_adam_noisy_dense_1_w_mu_v_read_readvariableop7savev2_adam_noisy_dense_1_w_sigma_v_read_readvariableop4savev2_adam_noisy_dense_1_b_mu_v_read_readvariableop7savev2_adam_noisy_dense_1_b_sigma_v_read_readvariableop4savev2_adam_noisy_dense_2_w_mu_v_read_readvariableop7savev2_adam_noisy_dense_2_w_sigma_v_read_readvariableop4savev2_adam_noisy_dense_2_b_mu_v_read_readvariableop7savev2_adam_noisy_dense_2_b_sigma_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::: : : : : : ::: : : : : : : ::::: : : : : : ::::::: : : : : : ::: 2(
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

: :$
 

_output_shapes

: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

: :$ 

_output_shapes

: : 

_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

: :$ 

_output_shapes

: : 

_output_shapes
:: 

_output_shapes
::$  

_output_shapes

::$! 

_output_shapes

:: "

_output_shapes
:: #

_output_shapes
::$$ 

_output_shapes

: :$% 

_output_shapes

: : &

_output_shapes
: : '

_output_shapes
: :$( 

_output_shapes

: :$) 

_output_shapes

: : *

_output_shapes
:: +

_output_shapes
::,

_output_shapes
: 
?
?
/__inference_DQN_Noisy_Net_layer_call_fn_3920752
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_DQN_Noisy_Net_layer_call_and_return_conditional_losses_39207372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
J__inference_DQN_Noisy_Net_layer_call_and_return_conditional_losses_3920857

inputs.
*noisy_dense_matmul_readvariableop_resource/
+noisy_dense_biasadd_readvariableop_resource0
,noisy_dense_1_matmul_readvariableop_resource1
-noisy_dense_1_biasadd_readvariableop_resource0
,noisy_dense_2_matmul_readvariableop_resource1
-noisy_dense_2_biasadd_readvariableop_resource
identity??"noisy_dense/BiasAdd/ReadVariableOp?!noisy_dense/MatMul/ReadVariableOp?$noisy_dense_1/BiasAdd/ReadVariableOp?#noisy_dense_1/MatMul/ReadVariableOp?$noisy_dense_2/BiasAdd/ReadVariableOp?#noisy_dense_2/MatMul/ReadVariableOp?
!noisy_dense/MatMul/ReadVariableOpReadVariableOp*noisy_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!noisy_dense/MatMul/ReadVariableOp?
noisy_dense/MatMulMatMulinputs)noisy_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
noisy_dense/MatMul?
"noisy_dense/BiasAdd/ReadVariableOpReadVariableOp+noisy_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"noisy_dense/BiasAdd/ReadVariableOp?
noisy_dense/BiasAddBiasAddnoisy_dense/MatMul:product:0*noisy_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
noisy_dense/BiasAdd|
noisy_dense/ReluRelunoisy_dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
noisy_dense/Relu?
#noisy_dense_1/MatMul/ReadVariableOpReadVariableOp,noisy_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#noisy_dense_1/MatMul/ReadVariableOp?
noisy_dense_1/MatMulMatMulnoisy_dense/Relu:activations:0+noisy_dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
noisy_dense_1/MatMul?
$noisy_dense_1/BiasAdd/ReadVariableOpReadVariableOp-noisy_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$noisy_dense_1/BiasAdd/ReadVariableOp?
noisy_dense_1/BiasAddBiasAddnoisy_dense_1/MatMul:product:0,noisy_dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
noisy_dense_1/BiasAdd?
noisy_dense_1/ReluRelunoisy_dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
noisy_dense_1/Relu?
#noisy_dense_2/MatMul/ReadVariableOpReadVariableOp,noisy_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#noisy_dense_2/MatMul/ReadVariableOp?
noisy_dense_2/MatMulMatMul noisy_dense_1/Relu:activations:0+noisy_dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
noisy_dense_2/MatMul?
$noisy_dense_2/BiasAdd/ReadVariableOpReadVariableOp-noisy_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$noisy_dense_2/BiasAdd/ReadVariableOp?
noisy_dense_2/BiasAddBiasAddnoisy_dense_2/MatMul:product:0,noisy_dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
noisy_dense_2/BiasAdd?
IdentityIdentitynoisy_dense_2/BiasAdd:output:0#^noisy_dense/BiasAdd/ReadVariableOp"^noisy_dense/MatMul/ReadVariableOp%^noisy_dense_1/BiasAdd/ReadVariableOp$^noisy_dense_1/MatMul/ReadVariableOp%^noisy_dense_2/BiasAdd/ReadVariableOp$^noisy_dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2H
"noisy_dense/BiasAdd/ReadVariableOp"noisy_dense/BiasAdd/ReadVariableOp2F
!noisy_dense/MatMul/ReadVariableOp!noisy_dense/MatMul/ReadVariableOp2L
$noisy_dense_1/BiasAdd/ReadVariableOp$noisy_dense_1/BiasAdd/ReadVariableOp2J
#noisy_dense_1/MatMul/ReadVariableOp#noisy_dense_1/MatMul/ReadVariableOp2L
$noisy_dense_2/BiasAdd/ReadVariableOp$noisy_dense_2/BiasAdd/ReadVariableOp2J
#noisy_dense_2/MatMul/ReadVariableOp#noisy_dense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
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
serving_default_input_1:0?????????A
noisy_dense_20
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?(
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
regularization_losses
trainable_variables
	variables
		keras_api


signatures
*Y&call_and_return_all_conditional_losses
Z_default_save_signature
[__call__"?%
_tf_keras_network?%{"class_name": "Functional", "name": "DQN_Noisy_Net", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "DQN_Noisy_Net", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "NoisyDense", "config": {"name": "noisy_dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.5, "maxval": 0.5, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.25}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.125}}}, "name": "noisy_dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "NoisyDense", "config": {"name": "noisy_dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.25, "maxval": 0.25, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.125}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.08838834764831843}}}, "name": "noisy_dense_1", "inbound_nodes": [[["noisy_dense", 0, 0, {}]]]}, {"class_name": "NoisyDense", "config": {"name": "noisy_dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.17677669529663687, "maxval": 0.17677669529663687, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.08838834764831843}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.35355339059327373}}}, "name": "noisy_dense_2", "inbound_nodes": [[["noisy_dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["noisy_dense_2", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 4]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "DQN_Noisy_Net", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "NoisyDense", "config": {"name": "noisy_dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.5, "maxval": 0.5, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.25}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.125}}}, "name": "noisy_dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "NoisyDense", "config": {"name": "noisy_dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.25, "maxval": 0.25, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.125}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.08838834764831843}}}, "name": "noisy_dense_1", "inbound_nodes": [[["noisy_dense", 0, 0, {}]]]}, {"class_name": "NoisyDense", "config": {"name": "noisy_dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.17677669529663687, "maxval": 0.17677669529663687, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.08838834764831843}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.35355339059327373}}}, "name": "noisy_dense_2", "inbound_nodes": [[["noisy_dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["noisy_dense_2", 0, 0]]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?
w_mu
w_sigma
b_mu
b_sigma

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
\__call__
*]&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "NoisyDense", "name": "noisy_dense", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "noisy_dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.5, "maxval": 0.5, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.25}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.125}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
?
w_mu
w_sigma
b_mu
b_sigma

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
^__call__
*_&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "NoisyDense", "name": "noisy_dense_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "noisy_dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.25, "maxval": 0.25, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.125}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.08838834764831843}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
?
w_mu
w_sigma
b_mu
b_sigma

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
`__call__
*a&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "NoisyDense", "name": "noisy_dense_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "noisy_dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "sigma": 0.5, "mu_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.17677669529663687, "maxval": 0.17677669529663687, "seed": 42}}, "sigma_w_initializer": {"class_name": "Constant", "config": {"value": 0.08838834764831843}}, "sigma_b_initializer": {"class_name": "Constant", "config": {"value": 0.35355339059327373}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?

#beta_1

$beta_2
	%decay
&learning_rate
'itermAmBmCmDmEmFmGmHmImJmKmLvMvNvOvPvQvRvSvTvUvVvWvX"
	optimizer
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
?
regularization_losses
trainable_variables
(metrics
	variables
)layer_regularization_losses

*layers
+non_trainable_variables
,layer_metrics
[__call__
Z_default_save_signature
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
,
bserving_default"
signature_map
": 2noisy_dense/w_mu
%:#2noisy_dense/w_sigma
:2noisy_dense/b_mu
!:2noisy_dense/b_sigma
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?
regularization_losses
-metrics
trainable_variables
	variables
.layer_regularization_losses

/layers
0non_trainable_variables
1layer_metrics
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
$:" 2noisy_dense_1/w_mu
':% 2noisy_dense_1/w_sigma
 : 2noisy_dense_1/b_mu
#:! 2noisy_dense_1/b_sigma
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?
regularization_losses
2metrics
trainable_variables
	variables
3layer_regularization_losses

4layers
5non_trainable_variables
6layer_metrics
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
$:" 2noisy_dense_2/w_mu
':% 2noisy_dense_2/w_sigma
 :2noisy_dense_2/b_mu
#:!2noisy_dense_2/b_sigma
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?
regularization_losses
7metrics
 trainable_variables
!	variables
8layer_regularization_losses

9layers
:non_trainable_variables
;layer_metrics
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
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
?
	=total
	>count
?	variables
@	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
=0
>1"
trackable_list_wrapper
-
?	variables"
_generic_user_object
':%2Adam/noisy_dense/w_mu/m
*:(2Adam/noisy_dense/w_sigma/m
#:!2Adam/noisy_dense/b_mu/m
&:$2Adam/noisy_dense/b_sigma/m
):' 2Adam/noisy_dense_1/w_mu/m
,:* 2Adam/noisy_dense_1/w_sigma/m
%:# 2Adam/noisy_dense_1/b_mu/m
(:& 2Adam/noisy_dense_1/b_sigma/m
):' 2Adam/noisy_dense_2/w_mu/m
,:* 2Adam/noisy_dense_2/w_sigma/m
%:#2Adam/noisy_dense_2/b_mu/m
(:&2Adam/noisy_dense_2/b_sigma/m
':%2Adam/noisy_dense/w_mu/v
*:(2Adam/noisy_dense/w_sigma/v
#:!2Adam/noisy_dense/b_mu/v
&:$2Adam/noisy_dense/b_sigma/v
):' 2Adam/noisy_dense_1/w_mu/v
,:* 2Adam/noisy_dense_1/w_sigma/v
%:# 2Adam/noisy_dense_1/b_mu/v
(:& 2Adam/noisy_dense_1/b_sigma/v
):' 2Adam/noisy_dense_2/w_mu/v
,:* 2Adam/noisy_dense_2/w_sigma/v
%:#2Adam/noisy_dense_2/b_mu/v
(:&2Adam/noisy_dense_2/b_sigma/v
?2?
J__inference_DQN_Noisy_Net_layer_call_and_return_conditional_losses_3920631
J__inference_DQN_Noisy_Net_layer_call_and_return_conditional_losses_3920857
J__inference_DQN_Noisy_Net_layer_call_and_return_conditional_losses_3920833
J__inference_DQN_Noisy_Net_layer_call_and_return_conditional_losses_3920612?
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
"__inference__wrapped_model_3920390?
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
/__inference_DQN_Noisy_Net_layer_call_fn_3920716
/__inference_DQN_Noisy_Net_layer_call_fn_3920752
/__inference_DQN_Noisy_Net_layer_call_fn_3920915
/__inference_DQN_Noisy_Net_layer_call_fn_3920898?
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
?2?
-__inference_noisy_dense_layer_call_fn_3920964
-__inference_noisy_dense_layer_call_fn_3920973?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_noisy_dense_layer_call_and_return_conditional_losses_3920947
H__inference_noisy_dense_layer_call_and_return_conditional_losses_3920936?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
/__inference_noisy_dense_1_layer_call_fn_3921022
/__inference_noisy_dense_1_layer_call_fn_3921031?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_noisy_dense_1_layer_call_and_return_conditional_losses_3920994
J__inference_noisy_dense_1_layer_call_and_return_conditional_losses_3921005?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
/__inference_noisy_dense_2_layer_call_fn_3921078
/__inference_noisy_dense_2_layer_call_fn_3921087?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_noisy_dense_2_layer_call_and_return_conditional_losses_3921061
J__inference_noisy_dense_2_layer_call_and_return_conditional_losses_3921051?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_signature_wrapper_3920779input_1"?
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
Const_5?
J__inference_DQN_Noisy_Net_layer_call_and_return_conditional_losses_3920612ucdefgh8?5
.?+
!?
input_1?????????
p

 
? "%?"
?
0?????????
? ?
J__inference_DQN_Noisy_Net_layer_call_and_return_conditional_losses_3920631i8?5
.?+
!?
input_1?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_DQN_Noisy_Net_layer_call_and_return_conditional_losses_3920833tcdefgh7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
J__inference_DQN_Noisy_Net_layer_call_and_return_conditional_losses_3920857h7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
/__inference_DQN_Noisy_Net_layer_call_fn_3920716hcdefgh8?5
.?+
!?
input_1?????????
p

 
? "???????????
/__inference_DQN_Noisy_Net_layer_call_fn_3920752\8?5
.?+
!?
input_1?????????
p 

 
? "???????????
/__inference_DQN_Noisy_Net_layer_call_fn_3920898gcdefgh7?4
-?*
 ?
inputs?????????
p

 
? "???????????
/__inference_DQN_Noisy_Net_layer_call_fn_3920915[7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
"__inference__wrapped_model_3920390y0?-
&?#
!?
input_1?????????
? "=?:
8
noisy_dense_2'?$
noisy_dense_2??????????
J__inference_noisy_dense_1_layer_call_and_return_conditional_losses_3920994def3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0????????? 
? ?
J__inference_noisy_dense_1_layer_call_and_return_conditional_losses_3921005`3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0????????? 
? ?
/__inference_noisy_dense_1_layer_call_fn_3921022Wef3?0
)?&
 ?
inputs?????????
p
? "?????????? ?
/__inference_noisy_dense_1_layer_call_fn_3921031S3?0
)?&
 ?
inputs?????????
p 
? "?????????? ?
J__inference_noisy_dense_2_layer_call_and_return_conditional_losses_3921051dgh3?0
)?&
 ?
inputs????????? 
p
? "%?"
?
0?????????
? ?
J__inference_noisy_dense_2_layer_call_and_return_conditional_losses_3921061`3?0
)?&
 ?
inputs????????? 
p 
? "%?"
?
0?????????
? ?
/__inference_noisy_dense_2_layer_call_fn_3921078Wgh3?0
)?&
 ?
inputs????????? 
p
? "???????????
/__inference_noisy_dense_2_layer_call_fn_3921087S3?0
)?&
 ?
inputs????????? 
p 
? "???????????
H__inference_noisy_dense_layer_call_and_return_conditional_losses_3920936dcd3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
H__inference_noisy_dense_layer_call_and_return_conditional_losses_3920947`3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
-__inference_noisy_dense_layer_call_fn_3920964Wcd3?0
)?&
 ?
inputs?????????
p
? "???????????
-__inference_noisy_dense_layer_call_fn_3920973S3?0
)?&
 ?
inputs?????????
p 
? "???????????
%__inference_signature_wrapper_3920779?;?8
? 
1?.
,
input_1!?
input_1?????????"=?:
8
noisy_dense_2'?$
noisy_dense_2?????????