###########################################################
### evaluation.attribution.conll.v5.pl
### Perl v5.18.4.
### Roser Morante, CLTL Lab, VU Amsterdam, May 2019
### Evaluate attribution sets
### Input format: two directories containing conll tab separated files one with with gold and another with system attribution sets
### Format input files:
### wsj_2400.xml	7	164	1	883,892	Meanwhile	meanwhile	RB	advmod	11	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
### wsj_2400.xml	7	165	2	892,893	,	,	,	_	_	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
### wsj_2400.xml	7	166	3	894,903	September	September	NNP	nn	5	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
### wsj_2400.xml	7	167	4	904,911	housing	housing	NN	nn	5	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
### wsj_2400.xml	7	168	5	912,918	starts	start	NNS	nsubjpass	11	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
### wsj_2400.xml	7	169	6	918,919	,	,	,	_	_	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
### wsj_2400.xml	7	170	7	920,923	due	due	JJ	amod	5	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
### wsj_2400.xml	7	171	8	924,933	Wednesday	Wednesday	NNP	tmod	7	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
### wsj_2400.xml	7	172	9	933,934	,	,	,	_	_	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
### wsj_2400.xml	7	173	10	935,938	are	be	VBP	auxpass	11	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ B-CUE-AT-12 _ _ _ _ _ _ _ _ _ _
### wsj_2400.xml	7	174	11	939,946	thought	think	VBN	root	0	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ I-CUE-AT-12 _ _ _ _ _ _ _ _ _ _
### wsj_2400.xml	7	175	12	947,949	to	to	TO	aux	14	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ B-CONTENT-AT-12 _ _ _ _ _ _ _ _ _ _
### wsj_2400.xml	7	176	13	950,954	have	have	VB	aux	14	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ I-CONTENT-AT-12 _ _ _ _ _ _ _ _ _ _
### wsj_2400.xml	7	177	14	955,961	inched	inch	VBN	xcomp	11	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ I-CONTENT-AT-12 _ _ _ _ _ _ _ _ _ _
### wsj_2400.xml	7	178	15	962,968	upward	upward	RB	advmod	14	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ I-CONTENT-AT-12 _ _ _ _ _ _ _ _ _ _
### wsj_2400.xml	7	179	16	968,969	.	.	.	_	_	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

### columns of gold and system files

### 1 name file
### 2 sentence number
### 3 token number in document (starts in 1)
### 4 token number in sentence (starts in 1)
### 5 begin offset, end offset
### 6 word form
### 7 lemma
### 8 part of speech
### 9 syntactic dependency label
### 10 syntactic dependency 
### 11-to-end as many columns as attribution sets there are; a column has the labels of one attribution set. The labels are: "_" if the token belongs to no attribution set; B/I-CUE-IDENTIFIER for cue, B/I-CONTENT-IDENTIFIER for content, B/I-SOURCE-IDENTIFIER for sources; B- means "beginning of", I- means "inside".
    

### File name format: files in the system directory have the same name as the gold files plus the extension provided in the -e option

### Examples: gold file is wsj_0024.xml and system files is  wsj_0024.xml.system;  extension is .system         
### Output: the scrips outputs evaluation resuls in terms of F scores
### Full match: all tokens in system output  need to have the same label as in gold
### Partial match: at least one token in system output needs to have the same label as in gold
### Overlap based: the overlap between gold and system is calculated as in Johansson and Moschitti 2013 Relational Features in Fine-Grained Opinion Analysis.

### Usage: evaluation.attribution.v5.pl [-h -v] -g <directory where gold files are> -s <directory where system files are> -o <output directory where evaluation per file is written, as well as files with fp, fn, tp> -e <extension of the system files>

#-------------------------------------------------------------------------------------------------

#!/usr/bin/perl
#use strict;
use warnings;
use Getopt::Std;

#---------------------------------------------------------
#---------------- declaring variables ---------------------

my $filename = "";
my $systemfilename = "";
my $file = "";
my $systemfile = "";


my %gold_cue = ();
my %gold_source = ();
my %gold_content = ();

my %system_cue = ();
my %system_source = ();
my %system_content = ();

my @line = ();
my $goldline = "";

my $match = 0;
my $partial_match = 0;
my @gold = ();
my @system = ();
my @intersection = ();

my @gold_content = ();
my @system_content = ();
my @intersection_content = ();

my @gold_source = ();
my @system_source = ();
my @intersection_source = ();

my @gold_cue = ();
my @system_cue = ();
my @intersection_cue = ();


my $total_content_gold = 0;
my $total_content_system = 0;
my $total_source_gold = 0;
my $total_source_system = 0;
my $total_cue_gold = 0;
my $total_cue_system = 0;
my $total_full_set_gold = 0;
my $total_full_set_system = 0;

my $total_nested_gold = 0;
my $total_content_nested_gold = 0;
my $total_source_nested_gold = 0;
my $total_cue_nested_gold = 0;

my $TP_content_only_strict = 0;
my $FP_content_only_strict = 0;
my $FN_content_only_strict = 0;

my $TP_cue_only_strict = 0;
my $FP_cue_only_strict = 0;
my $FN_cue_only_strict = 0;

my $NE_TP_cue_only_strict = 0;
my $NE_FN_cue_only_strict = 0;

my $NE_TP_source_only_strict = 0;
my $NE_FN_source_only_strict = 0;

my $NE_TP_content_only_strict = 0;
my $NE_FN_content_only_strict = 0;

my $NE_TP_full_set_strict = 0;
my $NE_FN_full_set_strict = 0;

my $perc_NE_TP_cue_only_strict = 0;
my $perc_NE_FN_cue_only_strict = 0;

my $perc_NE_TP_source_only_strict = 0;
my $perc_NE_FN_source_only_strict = 0;

my $perc_NE_TP_content_only_strict = 0;
my $perc_NE_FN_content_only_strict = 0;

my $perc_NE_TP_full_set_strict = 0;
my $perc_NE_FN_full_set_strict = 0;

my $TP_source_only_strict = 0;
my $FP_source_only_strict = 0;
my $FN_source_only_strict = 0;


my $TP_full_set_strict = 0;
my $FP_full_set_strict = 0;
my $FN_full_set_strict = 0;

my $precision_strict = 0;
my $recall_strict = 0;
my $f1_strict = 0;


my $TP_content_only_partial = 0;
my $FP_content_only_partial = 0;
my $FN_content_only_partial = 0;

my $TP_cue_only_partial = 0;
my $FP_cue_only_partial = 0;
my $FN_cue_only_partial = 0;

my $NE_TP_cue_only_partial = 0;
my $NE_FN_cue_only_partial = 0;

my $NE_TP_source_only_partial = 0;
my $NE_FN_source_only_partial = 0;

my $NE_TP_content_only_partial = 0;
my $NE_FN_content_only_partial = 0;

my $NE_TP_full_set_partial = 0;
my $NE_FN_full_set_partial = 0;

my $perc_NE_TP_cue_only_partial = 0;
my $perc_NE_FN_cue_only_partial = 0;

my $perc_NE_TP_source_only_partial = 0;
my $perc_NE_FN_source_only_partial = 0;

my $perc_NE_TP_content_only_partial = 0;
my $perc_NE_FN_content_only_partial = 0;

my $perc_NE_TP_full_set_partial = 0;
my $perc_NE_FN_full_set_partial = 0;

my $TP_source_only_partial = 0;
my $FP_source_only_partial = 0;
my $FN_source_only_partial = 0;


my $TP_full_set_partial = 0;
my $FP_full_set_partial = 0;
my $FN_full_set_partial = 0;

my $precision_partial = 0;
my $recall_partial = 0;
my $f1_partial = 0;

my $count_system = 0;

my $precision_content_partial = 0;
my $recall_content_partial = 0;
my $f1_content_partial = 0;

my $precision_content_strict = 0;
my $recall_content_strict = 0;
my $f1_content_strict = 0;

my $precision_source_partial = 0;
my $recall_source_partial = 0;
my $f1_source_partial = 0;

my $precision_source_strict = 0;
my $recall_source_strict = 0;
my $f1_source_strict = 0;

my $precision_cue_partial = 0;
my $recall_cue_partial = 0;
my $f1_cue_partial = 0;

my $precision_cue_strict = 0;
my $recall_cue_strict = 0;
my $f1_cue_strict = 0;

my $precision_full_set_partial = 0;
my $recall_full_set_partial = 0;
my $f1_full_set_partial = 0;

my $precision_full_set_strict = 0;
my $recall_full_set_strict = 0;
my $f1_full_set_strict = 0;

$size_system = 0;
$size_gold = 0;
$size_intersection = 0;

$span_coverage_content = 0;
$span_set_coverage_content = 0;
$precision_intersection_content = 0;
$recall_intersection_content = 0;

$span_coverage_source = 0;
$span_set_coverage_source = 0;
$precision_intersection_source = 0;
$recall_intersection_source = 0;

#-------------------------------------
#--------- options
#-------------------------------------

our ($opt_h, $opt_g, $opt_s, $opt_o, , $opt_e, $opt_v) ;

getopts("g:o:s:e:hv") ;


if ( defined $opt_h){

print<<'EOT';

      evaluation.attribution.conll.v5.pl
      Perl v5.18.4.
      Roser Morante, CLTL Lab, VU Amsterdam, May 2019
      Evaluate attribution sets
      Input format: two directories containing conll tab separated files one with with gold and another with system attribution sets
      Format input files:
      wsj_2400.xml	7	164	1	883,892	Meanwhile	meanwhile	RB	advmod	11	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
      wsj_2400.xml	7	165	2	892,893	,	,	,	_	_	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
      wsj_2400.xml	7	166	3	894,903	September	September	NNP	nn	5	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
      wsj_2400.xml	7	167	4	904,911	housing	housing	NN	nn	5	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
      wsj_2400.xml	7	168	5	912,918	starts	start	NNS	nsubjpass	11	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
      wsj_2400.xml	7	169	6	918,919	,	,	,	_	_	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
      wsj_2400.xml	7	170	7	920,923	due	due	JJ	amod	5	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
      wsj_2400.xml	7	171	8	924,933	Wednesday	Wednesday	NNP	tmod	7	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
      wsj_2400.xml	7	172	9	933,934	,	,	,	_	_	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
      wsj_2400.xml	7	173	10	935,938	are	be	VBP	auxpass	11	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ B-CUE-AT-12 _ _ _ _ _ _ _ _ _ _
      wsj_2400.xml	7	174	11	939,946	thought	think	VBN	root	0	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ I-CUE-AT-12 _ _ _ _ _ _ _ _ _ _
      wsj_2400.xml	7	175	12	947,949	to	to	TO	aux	14	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ B-CONTENT-AT-12 _ _ _ _ _ _ _ _ _ _
      wsj_2400.xml	7	176	13	950,954	have	have	VB	aux	14	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ I-CONTENT-AT-12 _ _ _ _ _ _ _ _ _ _
      wsj_2400.xml	7	177	14	955,961	inched	inch	VBN	xcomp	11	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ I-CONTENT-AT-12 _ _ _ _ _ _ _ _ _ _
      wsj_2400.xml	7	178	15	962,968	upward	upward	RB	advmod	14	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ I-CONTENT-AT-12 _ _ _ _ _ _ _ _ _ _
      wsj_2400.xml	7	179	16	968,969	.	.	.	_	_	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

      columns of gold and system files

      1 name file
      2 sentence number
      3 token number in document - starts in 1
      4 token number in sentence -starts in 1
      5 begin offset, end offset
      6 word form
      7 lemma
      8 part of speech
      9 syntactic dependency label
      10 syntactic dependency 
      11-to-end as many columns as attribution sets there are; a column has the labels of one attribution set. The labels are: _ if the token belongs to no attribution set; B/I-CUE-IDENTIFIER for cue, B/I-CONTENT-IDENTIFIER for content, B/I-SOURCE-IDENTIFIER for sources; B- means beginning of, I- means inside.
    

      File name format: files in the system directory have the same name as the system files plus the extension provided in the -e option

      Examples: gold file is wsj_0024.xml and system files is  wsj_0024.xml.system where base name is  wsj_0024.xml; extension is .system   
      
      Output: the scrips outputs evaluation resuls in terms of F scores

      Evaluation is performed at sequence level, not at token level. The script evaluated whether a sequence of tokens labelled as cue, source or content, is correctly predicted.
      Full match: all the sequence tokens in the system output  need to have the same label as in gold
      Partial match: at least one token in system output needs to have the same label as in gold
      To calculate the scores of  "content linked to source mention",  only partial match is required for content and source.
      Overlap based: the overlap between gold and system is calculated as in Johansson and Moschitti 2013 Relational Features in Fine-Grained Opinion Analysis.

      Usage: evaluation.attribution.v5.pl [-h -v] -g <directory where gold files are> -s <directory where system files are> -o <output directory where evaluation per file is written, as well as files with fp, fn, tp> -e <extension of the system files>

     Optional parameter:

     -h : help
     -v : verbose (not fully implemented yet)

EOT
  exit;
}

#----------------------------------------------------
#------------ calculate intersection of arrays ------

sub intersect(\@\@) {
#  print "intersect in:\n\t0:  @{$_[0]} \n\t1:  @{$_[1]}\n";
	my %e = map { $_ => undef } @{$_[0]};
#  print "intersect out: " . grep {exists( $e{$_})} @{$_[1]}. "\n\n";
	return grep { exists( $e{$_} ) } @{$_[1]};
}


#------------------------------------------
#------------- MAIN -----------------------

$directory_gold = $opt_g;
$directory_system = $opt_s;
$out_directory = $opt_o;
$extension = $opt_e;


opendir (DIRGOLD, "$directory_gold") or die $!;
opendir (DIRSYSTEM, "$directory_system") or die $!;

while ($file = readdir(DIRGOLD)){
  if ($file eq '.' or $file eq '..' or $file eq '.DS_Store') {
    next;
  }

#  if ($file =~ /^(.+).gold$/){    
#   $filename = $1;
  $filename = $file;
    if ( defined $opt_v){
      print STDERR "gold file is $filename\n";
    }
    
    open(OUTFILE, ">$out_directory\/$filename.eval.out")  || die "Cannot open out file\n"; 
    open(GOLDFILE,"$directory_gold/$file") or die "Cannot open gold file\n";
    
    
    $systemfilename = "$filename" . "$extension";
    
    if ( defined $opt_v){
      print STDERR "system file is $directory_system$systemfilename\n";
    }
    open(SYSTEMFILE, "$directory_system/$systemfilename") or die "Cannot open system file\n";
    
    if (defined $opt_v) {
      print STDERR "gold file: $directory_gold/$file\n";
      print STDERR "system file: $directory_system/$systemfilename\n";
    }
    
    
    
    #------------------------------------------------------------
    #----- read files and store attribution information in hashes
    #------------------------------------------------------------
    read_gold_file();
    read_system_file();


    #-------------------------------------------------------------------------------------
    #-- uncomment code to print identifiers of tokens that belong to each attribution set
    #-------------------------------------------------------------------------------------
    #print_tokens_per_set();

    
    if (%gold_content){
      $size = keys %gold_content;
      $total_content_gold = $size + $total_content_gold;
    }
    if (%system_content){
      $size = keys %system_content;
      $total_content_system = $size + $total_content_system;
    }
    if (%gold_source){
      $size  = keys %gold_source;
      $total_source_gold = $size + $total_source_gold;
    }
    if (%system_source){
      $size = keys %system_source;
      $total_source_system = $size + $total_source_system;
    }
    if (%gold_cue){
      $size = keys %gold_cue;
      $total_cue_gold = $size  + $total_cue_gold;
    }
    if (%system_cue){
      $size  = keys %system_cue;
      $total_cue_system = $size + $total_cue_system;
    }
    if (%gold_content){
      $size = keys %gold_content;
      $total_full_set_gold = $size + $total_full_set_gold;
    }
    if (%system_content){
      $size = keys %system_content;
      $total_full_set_system = $size + $total_full_set_system;
    }
    
    
    #------------------------------------------------------------
    #----- evaluate
    #------------------------------------------------------------

    evaluate2();
    
	 
#  } #file is .gold 


  close(SYSTEMFILE);
  close(GOLDFILE);
  
  calculate_results();
  print_results_perfile();

  close(OUTFILE);
    
}##while over gold files

closedir(DIRGOLD);
closedir(DIRSYSTEM);

calculate_results();
print_results();


#-----------------------------------
sub read_gold_file {
#-----------------------------------

# Reading system and gold files in conll format
#wsj_2400.xml	6	143	7	745,749	much	much	JJ	dep	11	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ I-CONTENT-AT-11 _ I-CONTENT-NE-23 _ _ _ _ _ _


@line = ();

$goldline = "";

%gold_cue = ();
%gold_source = ();
%gold_content = ();


%system_cue = ();
%system_source = ();
%system_content = ();

while (<GOLDFILE>){
  $goldline = $_;
  chomp $goldline;
  @line = ();
  @attrLabels = ();
  ### check whether line is not blank
  ### create hashes with attribution sets
  ### counting attribution components 
  @line = ();
  @atrLabels = ();
  if ($goldline =~ /^.+\t.+\t.+$/){
    @line = split /\t/, $goldline;
    @attrLabels = split / /, $line[10];
    for ($i=0;$i<=$#attrLabels;$i++){
      if ($attrLabels[$i] =~ /^.+-CUE-(.+)$/){
	$id = $1;
	#$gold_cue{ "$id" } = [];
	push @{ $gold_cue{"$id"} }, $line[2];
      } elsif ($attrLabels[$i] =~ /^.+-CONTENT-(.+)$/){
	$id = $1;
	#$gold_content{ "$id" } = [];
	push @{ $gold_content{"$id"} }, $line[2];
      } elsif ($attrLabels[$i] =~ /^.+-SOURCE-(.+)$/){
	$id = $1;
	#$gold_source{ "$id" } = [];
	push @{ $gold_source{"$id"} }, $line[2];
      }
    }#for over $i
    
  } #if line is not blank
  
}#while GOLDFILE

}

#-----------------------------------
sub read_system_file {
#-----------------------------------

# Reading system and gold files in conll format
#wsj_2400.xml	6	143	7	745,749	much	much	JJ	dep	11	_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ I-CONTENT-AT-11 _ I-CONTENT-NE-23 _ _ _ _ _ _


my $systemline = "";

while (<SYSTEMFILE>){
  $systemline = $_;
  chomp $systemline;
  @line = ();
  @attrLabels = ();
  ### check whether line is not blank
  ### create hashes with attribution sets
  ### counting attribution components 
  @line = ();
  @atrLabels = ();
  if ($systemline =~ /^.+\t.+\t.+$/){
    @line = split /\t/, $systemline;
    @attrLabels = split / /, $line[10];
    for ($i=0;$i<=$#attrLabels;$i++){
      if ($attrLabels[$i] =~ /^.+-CUE-(.+)$/){
	$id = $1;
	#$system_cue{ "$id" } = [];
	push @{ $system_cue{"$id"} }, $line[2];
      } elsif ($attrLabels[$i] =~ /^.+-CONTENT-(.+)$/){
	$id = $1;
	#$system_content{ "$id" } = [];
	push @{ $system_content{"$id"} }, $line[2];
      } elsif ($attrLabels[$i] =~ /^.+-SOURCE-(.+)$/){
	$id = $1;
	#$system_source{ "$id" } = [];
	push @{ $system_source{"$id"} }, $line[2];
      }
    }#for over $i
    
   } #if line is not blank
  
}#while SYSTEMFILE
  
}#closes read_system_file


#-------------------------------------------------
sub print_tokens_per_set {
#-------------------------------------------------

#-- print identifiers of tokens that belong to each attribution set

 print OUTFILE "================  G O L D  H A S H E S  =========================\n";

 for $bla ( keys %gold_cue ) {
   print  OUTFILE "$bla Cues";
   foreach $i (0 .. $#{ $gold_cue{$bla} } ) {
     print OUTFILE "$i = $gold_cue{$bla}[$i] ";
   }
   print  OUTFILE "\n";
 }

 print OUTFILE "=========================\n";

 for $bla ( keys %gold_content ) {
   print  OUTFILE "$bla Content: ";
   foreach $i (0 .. $#{ $gold_content{$bla} } ) {       
     print OUTFILE "$i = $gold_content{$bla}[$i] ";
   }
   print  OUTFILE "\n";
 }


 print OUTFILE "=========================\n";

 for $bla ( keys %gold_source ) {
   print  OUTFILE "$bla Source: ";
   foreach $i (0 .. $#{ $gold_source{$bla} } ) {       
     print OUTFILE "$i = $gold_source{$bla}[$i] ";
   }
   print  OUTFILE "\n";
 }

	

  print OUTFILE "================  S Y S T E M  H A S H E S  =========================\n";
	
  	   for $bla ( keys %system_cue ) {
  	       print  OUTFILE "$bla Cues";
  	       foreach $i (0 .. $#{ $system_cue{$bla} } ) {
  	           print OUTFILE "$i = $system_cue{$bla}[$i] ";
  	       }
  	       print  OUTFILE "\n";
  	   }
	
  	  print OUTFILE "=========================\n";
	
  	  for $bla ( keys %system_content ) {
  	       print  OUTFILE "$bla Content: ";
  	       foreach $i (0 .. $#{ $system_content{$bla} } ) {       
  	           print OUTFILE "$i = $system_content{$bla}[$i] ";
  	       }
  	       print  OUTFILE "\n";
  	   }
	
	
  	  print OUTFILE "=========================\n";
	
  	  for $bla ( keys %system_source ) {
  	       print  OUTFILE "$bla Source: ";
  	       foreach $i (0 .. $#{ $system_source{$bla} } ) {       
  	           print OUTFILE "$i = $system_source{$bla}[$i] ";
  	       }
  	       print  OUTFILE "\n";
  	   }
	
	
}#closes function



#-------------------------------------------------
sub evaluate2 {
#-------------------------------------------------

  calculate_span_coverage_metrics();


#---------------------------------------------------
#-- calculate TP, FN, FP
#---------------------------------------------------

#----------------------------------------------
#- C O N T E N T
#----------------------------------------------
#- iterate over gold CONTENT sets
#- looking for TP, FN
#----------------------------------------------

  # create temporal hashes because they will be modified when span matches are found
  %tmp_system_content = %system_content;
  %tmp_gold_content = %gold_content;
  
  
  # iterate over gold keys
  for $g ( keys %gold_content ) {
    @gold = ();
    @gold = @{ $gold_content{$g} };
    
    #maximum number of tokens in the intersection
    $max_intersection = 0;
    $id_max_intersection = "_";
    @max_intersection = ();
    @intersection = ();
    
    #iterate over system contents
    for $s ( keys %system_content ) {
      @system = ();
      if (defined $tmp_system_content{$s}){
	@system = @{ $tmp_system_content{$s} };
	
	@intersection = ();
	@intersection = intersect(@gold, @system);
	# if (($#intersection > $#gold) || ($#intersection > $#system)) {
	#   print "ALERT: intersection cannot be bigger than gold or system\n";
	#   print "gold [ $#gold]: @gold\n";
	#   print "system [ $#system ]: @system\n";
	#   print "isect [ $#intersection ]: @intersection\n";
	# }
	
	
	if ( @intersection){
	  # update information about the sequence that matches
	  if ($#intersection == $#gold && $#intersection == $#system){
	    $max_intersection = $#intersection + 1;
	    $id_max_intersection = "$s"; 
	    @max_intersection = ();
	    @max_intersection = @intersection;
	  }elsif (($#intersection + 1) > $max_intersection){
	    $max_intersection = $#intersection + 1;
	    $id_max_intersection = "$s"; 
	    @max_intersection = ();
	    @max_intersection = @intersection;
	    
	  }
	} 
	
      }#if defined system
    }#for system 
    
    # if an intersection has been found we eliminate that sequence from system sequences in the temporal hash
    if ($id_max_intersection ne "_"){
      $tmp_system_content{$id_max_intersection} = undef;
    }
    
    


    #---------------------------------------------------
    #- info needed to check whether content and source are linked
    #---------------------------------------------------

    my @gold_cue = ();
    if (exists $gold_cue{$g}){
      @gold_cue = @{ $gold_cue{$g} };
    }
    my @gold_source = ();
    if (exists $gold_source{$g}){
      @gold_source = @{ $gold_source{$g} };
    }
    
    my @system_source = ();
    if (exists $system_source{$id_max_intersection}){
      @system_source = @{$system_source{$id_max_intersection}};
    }
    my @system_cue = ();
    if (exists $system_cue{$id_max_intersection}){
      @system_cue = @{$system_cue{$id_max_intersection}};
    }
    
    @intersection_source = ();
    @intersection_source = intersect(@gold_source, @system_source);
    @intersection_cue = ();
    @intersection_cue = intersect(@gold_cue, @system_cue);
    
    
    #---------------------------
    #---- perfect content match
    #---------------------------
    ## calculate TP, FN
    ## system and gold are the same
    if ( @max_intersection && $#max_intersection == $#gold && $#gold == $#{$system_content{$id_max_intersection}}){
      $TP_content_only_strict++;
      $TP_content_only_partial++;
      ## if there is a source, there has to be intersection of source and content for the set to be correct
      if (@gold_source){
	if (@max_intersection &&  @intersection_source) {
	  $TP_full_set_partial++;
	} else {
	  $FN_full_set_partial++;	
	  ## if there is no gold source, we do not need to check system source
	}
      } else{
	$TP_full_set_partial++;
      }
      
      #--------------------------
      #---- partial content match
      #---------------------------
      ## partial overlap system and gold
    } elsif ( @max_intersection){
      $TP_content_only_partial++;
      $FN_content_only_strict++;
      
      ## if there is a source, there has to be intersection of source and content for the set to be correct
      if (@gold_source){
	if (@max_intersection &&  @intersection_source) {
	  $TP_full_set_partial++;
	} else {
	  
	  $FN_full_set_partial++;	
	  ## if there is no gold source, we do not need to check system source
	}
      } else{
	$TP_full_set_partial++;
      }
      
      #--------------------------
      #---- no content match
      #---------------------------
      ## no overlap system and gold
    } elsif(!( @max_intersection)){
      
      $FN_content_only_strict++;
      $FN_content_only_partial++;
      $FN_full_set_partial++;
    }


  }#for iterating over gold


  #####################################
  ## iterate over system CONTENT sets
  ## looking for FP
  #####################################
  
  # iterate over system
  for $s ( keys %system_content ) {
    $match = 0;
    @system = ();
    @system = @{ $system_content{$s} };
    
    $max_intersection = 0;
    $id_max_intersection = "_";
    @max_intersection = ();
    @intersection = ();
    
    #iterate over gold
    for $g ( keys %tmp_gold_content ) {
      @gold = ();
      if (defined $tmp_gold_content{$g}){
	@gold = @{ $tmp_gold_content{$g} };
	@intersection = ();
	@intersection = intersect(@gold, @system);
	
	if ( @intersection){
	  if ($#intersection == $#gold && $#intersection == $#system){
	    $max_intersection = $#intersection + 1;
	    $id_max_intersection = "$g"; 
	    @max_intersection = ();
	    @max_intersection = @intersection;
	  } elsif (($#intersection + 1) > $max_intersection){
	    $max_intersection = $#intersection + 1;
	    $id_max_intersection = "$g"; 
	    @max_intersection = ();
	    @max_intersection = @intersection;
	  }
	  
	}
	
      }#if defined gold
    }#for gold
    
    if ($id_max_intersection ne "_"){
      $tmp_gold_content{$id_max_intersection} = undef;
    }
    
    #---------------------------------------------------
    #- info needed to check whether content and source are linked
    #---------------------------------------------------
  
    @system_cue = ();
    if (exists $system_cue{$s} ) {
      @system_cue = @{ $system_cue{$s} };
    }
    @system_source = ();
    if (exists  $system_source{$s} ){
      @system_source = @{ $system_source{$s} };
    }
    @gold_source = ();
    if (exists $gold_source{$id_max_intersection}){
      @gold_source = @{$gold_source{$id_max_intersection}};
    }
    @gold_cue = ();
    if (exists $gold_cue{$id_max_intersection}){
      @gold_cue = @{$gold_cue{$id_max_intersection}};
    }
    @intersection_source = ();
    @intersection_source = intersect(@gold_source, @system_source);
    
    
    ## some overlap between system and gold but not same span
    if ( @max_intersection && !(compare_arrays(\@system,\@{$gold_content{$id_max_intersection}}))){
      $FP_content_only_strict++;
      #some overlap system and gold, then we check whehter exists gold source and whether there is intersection between sources
    } elsif ( @system_source && @max_intersection && !(@intersection_source)){
      $FP_full_set_partial++;
      ## no overlap system and gold
    } elsif(!( @max_intersection)){
      $FP_content_only_strict++;
      $FP_content_only_partial++;
      $FP_full_set_partial++;
    }
  
  
  }#for system
 

#----------------------------------------------
#- S O U R C E
#----------------------------------------------
  
#####################################
## iterate over gold SOURCE sets
## looking for TP, FN
#####################################

# create temporal hashes because they will be modified
%tmp_system_source = %system_source;
%tmp_gold_source = %gold_source;

# iterate over gold keys
for $g ( keys %gold_source ) {
  @gold = ();
  @gold = @{ $gold_source{$g} };
  
  #maximum number of tokens in the intersection
  $max_intersection = 0;
  $id_max_intersection = "_";
  @max_intersection = ();
  @intersection = ();
  
  #iterate over system sources
  for $s ( keys %tmp_system_source ) {
    @system = ();
    if (defined $tmp_system_source{$s}){
      @system = @{ $tmp_system_source{$s} };
      @intersection = ();
      @intersection = intersect(@gold, @system);
      # update information about the scope that matches
      if ( @intersection){
	#print "I am in defined intersection @intersection\n";
	if ($#intersection == $#gold && $#intersection == $#system){
	  $max_intersection = $#intersection + 1;
	  $id_max_intersection = "$g"; 
	  @max_intersection = ();
	  @max_intersection = @intersection;
	} elsif (($#intersection + 1) > $max_intersection){
	  $max_intersection = $#intersection + 1;
	  $id_max_intersection = "$s"; 
	  @max_intersection = ();
	  @max_intersection = @intersection;
	}
      } 
    }#if defined system
  }#for system 
  
  
  # if an intersection has been found we eliminate that scope from system scopes in the temporal hash
  if ($id_max_intersection ne "_"){
    $tmp_system_source{$id_max_intersection} = undef;
  }
   
  ## calculate TP, FN
  ## system and gold are the same
  if ( @max_intersection && $#max_intersection == $#gold && $#gold == $#{$system_source{$id_max_intersection}}){
    $TP_source_only_strict++;
    $TP_source_only_partial++;
    ## partial overlap system and gold
  } elsif ( @max_intersection){
    $TP_source_only_partial++;
    $FN_source_only_strict++;
    ## no overlap system and gold
  } elsif(!( @max_intersection)){
    $FN_source_only_strict++;
    $FN_source_only_partial++;
  }
  
  
}#for gold


#####################################
## iterate over system SOURCE sets
## looking for FP
#####################################

# iterate over system
for $s ( keys %system_source ) {
  @system = ();
  @system = @{ $system_source{$s} };
  
  $max_intersection = 0;
  $id_max_intersection = "_";
  @max_intersection = ();
  @intersection = ();
  
  #iterate over gold
  for $g ( keys %tmp_gold_source ) {
    @gold = ();
    if (defined $tmp_gold_source{$g}){
      @gold = @{ $tmp_gold_source{$g} };
      @intersection = ();
      @intersection = intersect(@gold, @system);
      if ( @intersection){
	if ($#intersection == $#gold && $#intersection == $#system){
	  $max_intersection = $#intersection + 1;
	  $id_max_intersection = "$g"; 
	  @max_intersection = ();
	  @max_intersection = @intersection;
	}elsif (($#intersection + 1) > $max_intersection){
	  $max_intersection = $#intersection + 1;
	  $id_max_intersection = "$g"; 
	  @max_intersection = ();
	  @max_intersection = @intersection;
	}
      }
      
    }#if defined gold
  }#for gold
  
  if ($id_max_intersection ne "_"){
    $tmp_gold_source{$id_max_intersection} = undef;
  }
  
  
  if ( @max_intersection && !(compare_arrays(\@system,\@{$gold_source{$id_max_intersection}}))){
    $FP_source_only_strict++;
    ## no overlap system and gold
  } elsif(!( @max_intersection)){
    $FP_source_only_strict++;
    $FP_source_only_partial++;
  }
  
  
}#for system
  

#----------------------------------------------
#- C U E
#----------------------------------------------


#####################################
## iterate over gold CUE sets
## looking for TP, FN
#####################################

# create temporal hashes because they will be modified
%tmp_system_cue = %system_cue;
%tmp_gold_cue = %gold_cue;
  
# iterate over gold keys
for $g ( keys %gold_cue ) {
  @gold = ();
  @gold = @{ $gold_cue{$g} };
  
  #maximum number of tokens in the intersection
  $max_intersection = 0;
  $id_max_intersection = "_";
  @max_intersection = ();
  @intersection = ();
  
  #iterate over system cues
  for $s ( keys %tmp_system_cue ) {
    @system = ();
    if (defined $tmp_system_cue{$s}){
      @system = @{ $tmp_system_cue{$s} };
      @intersection = ();
      @intersection = intersect(@gold, @system);
      # update information about the scope that matches
      if ( @intersection){
	if ($#intersection == $#gold && $#intersection == $#system){
	  $max_intersection = $#intersection + 1;
	  $id_max_intersection = "$g"; 
	  @max_intersection = ();
	  @max_intersection = @intersection;
	} elsif (($#intersection + 1) > $max_intersection){
	  $max_intersection = $#intersection + 1;
	  $id_max_intersection = "$s"; 
	  @max_intersection = ();
	  @max_intersection = @intersection;
	}
      }
    }#if defined system
  }#for system 
  
  
  # if an intersection has been found we eliminate that scope from system scopes in the temporal hash
  if ($id_max_intersection ne "_"){
    $tmp_system_cue{$id_max_intersection} = undef;
  }
  
 
  ## calculate TP, FN
  ## system and gold are the same
  if ( @max_intersection && $#max_intersection == $#gold && $#gold == $#{$system_cue{$id_max_intersection}}){
    $TP_cue_only_strict++;
    $TP_cue_only_partial++;
    ## partial overlap system and gold
  } elsif ( @max_intersection){
    $TP_cue_only_partial++;
    $FN_cue_only_strict++;
    ## no overlap system and gold
  } elsif(!( @max_intersection)){
    $FN_cue_only_strict++;
    $FN_cue_only_partial++;
  }
  
  
}#for gold


#####################################
## iterate over system CUE sets
## looking for FP
#####################################

# iterate over system
for $s ( keys %system_cue ) {
  @system = ();
  @system = @{ $system_cue{$s} };
  
  $max_intersection = 0;
  $id_max_intersection = "_";
  @max_intersection = ();
  @intersection = ();
  
  #iterate over gold
  for $g ( keys %tmp_gold_cue ) {
    @gold = ();
    if (defined $tmp_gold_cue{$g}){
      @gold = @{ $tmp_gold_cue{$g} };
      @intersection = ();
      @intersection = intersect(@gold, @system);
      if ( @intersection){
	if ($#intersection == $#gold && $#intersection == $#system){
	  $max_intersection = $#intersection + 1;
	  $id_max_intersection = "$g"; 
	  @max_intersection = ();
	  @max_intersection = @intersection;
	}elsif (($#intersection + 1) > $max_intersection){
	  $max_intersection = $#intersection + 1;
	  $id_max_intersection = "$g"; 
	  @max_intersection = ();
	  @max_intersection = @intersection;
	}
      }
      
    }#if defined gold
  }#for gold
  
  if ($id_max_intersection ne "_"){
    $tmp_gold_cue{$id_max_intersection} = undef;
  }
  
 
  
  if ( @max_intersection && !(compare_arrays(\@system,\@{$gold_cue{$id_max_intersection}}))){
    $FP_cue_only_strict++;
    ## no overlap system and gold
  } elsif(!( @max_intersection)){
    $FP_cue_only_strict++;
    $FP_cue_only_partial++;
  }
  

}#for system
  
  
}### finishes function



#-----------------------------------------------
sub calculate_span_coverage_metrics {
#-----------------------------------------------


#---------------------------------------------------
#-- calculate intersection evaluation metrics
#-- calculate span coverage and span set coverage (Johansson and Moschitti 2013)
#---------------------------------------------------


#---------
#- content
#---------

for $g ( keys %gold_content ) {

  @gold = ();
  @gold = @{ $gold_content{$g} };    
  
  for $s ( keys %system_content ) {
    
    @system = ();
    
    @system = @{ $system_content{$s} };
    @intersection = ();
    @intersection = intersect(@gold, @system);
    
    if ( @system){
      $size_system = $#system + 1;
    } else {
      $size_system = 0;
    }
    if ( @gold){
      $size_gold = $#gold + 1;
    } else {
      $size_gold = 0;
    }
    if ( @intersection){
      $size_intersection = $#intersection + 1;
    } else {
      $size_intersection = 0;
    }
    
    #print "size intersection $size_intersection\n";
    #print "size system $size_system\n";
    #print "size gold $size_gold\n";
   
    $span_coverage_content = $size_intersection / $size_gold;
    $span_set_coverage_content = $span_set_coverage_content + $span_coverage_content;
    #print "coverage $span_coverage_content\n";
    #print "coverage set  $span_coverage_content\n";
    
  } #for system
    
} #for gold

#------------------
#- source
#------------------

for $g ( keys %gold_source ) {

  @gold = ();
  @gold = @{ $gold_source{$g} };    
  
  for $s ( keys %system_source ) {
    
    @system = ();
    
    @system = @{ $system_source{$s} };
    @intersection = ();
    @intersection = intersect(@gold, @system);
    
    if ( @system){
      $size_system = $#system + 1;
    } else {
      $size_system = 0;
    }
    if ( @gold){
      $size_gold = $#gold + 1;
    } else {
      $size_gold = 0;
    }
    if ( @intersection){
      $size_intersection = $#intersection + 1;
    } else {
      $size_intersection = 0;
    }
    
    #print "intersection is @intersection\n";
    #print "size intersection $size_intersection\n";
    #print "size system $size_system\n";
    #print "size gold $size_gold\n";
   
    $span_coverage_source = $size_intersection / $size_gold;
    $span_set_coverage_source = $span_set_coverage_source + $span_coverage_source;
    #print "coverage $span_coverage_source\n";
    #print "coverage set  $span_coverage_source\n";
    
  } #for system
    
} #for gold

}#finishes function



#------------------------------------------------------
sub calculate_results{
#------------------------------------------------------
  
#------ CONTENT ONLY -------------


if ($TP_content_only_strict + $FP_content_only_strict){
   $precision_content_strict = sprintf("%.2f",($TP_content_only_strict / ($TP_content_only_strict + $FP_content_only_strict)) * 100);
 } else {
   $precision_content_strict = sprintf("%.2f",0.00);
 }

if ($TP_content_only_strict + $FN_content_only_strict){
  $recall_content_strict =  sprintf("%.2f",($TP_content_only_strict / ($TP_content_only_strict + $FN_content_only_strict)) * 100);
} else {
  $recall_content_strict = sprintf("%.2f",0.00);
}

if ($precision_content_strict + $recall_content_strict){
  $f1_content_strict =   sprintf("%.2f",(2 * $precision_content_strict * $recall_content_strict) / ($precision_content_strict + $recall_content_strict));
} else {
  $f1_content_strict = sprintf("%.2f",0.00);
}


if ($TP_content_only_partial + $FP_content_only_partial){
   $precision_content_partial = sprintf("%.2f",($TP_content_only_partial / ($TP_content_only_partial + $FP_content_only_partial)) * 100);
 } else {
   $precision_content_partial = sprintf("%.2f",0.00);
 }

if ($TP_content_only_partial + $FN_content_only_partial){
  $recall_content_partial =  sprintf("%.2f",($TP_content_only_partial / ($TP_content_only_partial + $FN_content_only_partial)) * 100);
} else {
  $recall_content_partial = sprintf("%.2f",0.00);
}

if ($precision_content_partial + $recall_content_partial){
  $f1_content_partial =   sprintf("%.2f",(2 * $precision_content_partial * $recall_content_partial) / ($precision_content_partial + $recall_content_partial));
} else {
  $f1_content_partial = sprintf("%.2f",0.00);
}






#--------- SOURCE ONLY  ----------------------

if ($TP_source_only_strict + $FP_source_only_strict){
   $precision_source_strict = sprintf("%.2f",($TP_source_only_strict / ($TP_source_only_strict + $FP_source_only_strict)) * 100);
 } else {
   $precision_source_strict = sprintf("%.2f",0.00);
 }

if ($TP_source_only_strict + $FN_source_only_strict){
  $recall_source_strict =  sprintf("%.2f",($TP_source_only_strict / ($TP_source_only_strict + $FN_source_only_strict)) * 100);
} else {
  $recall_source_strict = sprintf("%.2f",0.00);
}

if ($precision_source_strict + $recall_source_strict){
  $f1_source_strict =   sprintf("%.2f",(2 * $precision_source_strict * $recall_source_strict) / ($precision_source_strict + $recall_source_strict));
} else {
  $f1_source_strict = sprintf("%.2f",0.00);
}


if ($TP_source_only_partial + $FP_source_only_partial){
   $precision_source_partial = sprintf("%.2f",($TP_source_only_partial / ($TP_source_only_partial + $FP_source_only_partial)) * 100);
 } else {
   $precision_source_partial = sprintf("%.2f",0.00);
 }

if ($TP_source_only_partial + $FN_source_only_partial){
  $recall_source_partial =  sprintf("%.2f",($TP_source_only_partial / ($TP_source_only_partial + $FN_source_only_partial)) * 100);
} else {
  $recall_source_partial = sprintf("%.2f",0.00);
}

if ($precision_source_partial + $recall_source_partial){
  $f1_source_partial =   sprintf("%.2f",(2 * $precision_source_partial * $recall_source_partial) / ($precision_source_partial + $recall_source_partial));
} else {
  $f1_source_partial = sprintf("%.2f",0.00);
}

#------------------ CUE ONLY -----------------------


if ($TP_cue_only_strict + $FP_cue_only_strict){
   $precision_cue_strict = sprintf("%.2f",($TP_cue_only_strict / ($TP_cue_only_strict + $FP_cue_only_strict)) * 100);
 } else {
   $precision_cue_strict = sprintf("%.2f",0.00);
 }

if ($TP_cue_only_strict + $FN_cue_only_strict){
  $recall_cue_strict =  sprintf("%.2f",($TP_cue_only_strict / ($TP_cue_only_strict + $FN_cue_only_strict)) * 100);
} else {
  $recall_cue_strict = sprintf("%.2f",0.00);
}

if ($precision_cue_strict + $recall_cue_strict){
  $f1_cue_strict =   sprintf("%.2f",(2 * $precision_cue_strict * $recall_cue_strict) / ($precision_cue_strict + $recall_cue_strict));
} else {
  $f1_cue_strict = sprintf("%.2f",0.00);
}


if ($TP_cue_only_partial + $FP_cue_only_partial){
   $precision_cue_partial = sprintf("%.2f",($TP_cue_only_partial / ($TP_cue_only_partial + $FP_cue_only_partial)) * 100);
 } else {
   $precision_cue_partial = sprintf("%.2f",0.00);
 }

if ($TP_cue_only_partial + $FN_cue_only_partial){
  $recall_cue_partial =  sprintf("%.2f",($TP_cue_only_partial / ($TP_cue_only_partial + $FN_cue_only_partial)) * 100);
} else {
  $recall_cue_partial = sprintf("%.2f",0.00);
}

if ($precision_cue_partial + $recall_cue_partial){
  $f1_cue_partial =   sprintf("%.2f",(2 * $precision_cue_partial * $recall_cue_partial) / ($precision_cue_partial + $recall_cue_partial));
} else {
  $f1_cue_partial = sprintf("%.2f",0.00);
}


#-------------------- FULL SET -------------------------------------

# if ($TP_full_set_strict + $FP_full_set_strict){
#    $precision_full_set_strict = sprintf("%.2f",($TP_full_set_strict / ($TP_full_set_strict + $FP_full_set_strict)) * 100);
#  } else {
#    $precision_full_set_strict = sprintf("%.2f",0.00);
#  }

# if ($TP_full_set_strict + $FN_full_set_strict){
#   $recall_full_set_strict =  sprintf("%.2f",($TP_full_set_strict / ($TP_full_set_strict + $FN_full_set_strict)) * 100);
# } else {
#   $recall_full_set_strict = sprintf("%.2f",0.00);
# }

# if ($precision_full_set_strict + $recall_full_set_strict){
#   $f1_full_set_strict =   sprintf("%.2f",(2 * $precision_full_set_strict * $recall_full_set_strict) / ($precision_full_set_strict + $recall_full_set_strict));
# } else {
#   $f1_full_set_strict = sprintf("%.2f",0.00);
# }


if ($TP_full_set_partial + $FP_full_set_partial){
   $precision_full_set_partial = sprintf("%.2f",($TP_full_set_partial / ($TP_full_set_partial + $FP_full_set_partial)) * 100);
 } else {
   $precision_full_set_partial = sprintf("%.2f",0.00);
 }

if ($TP_full_set_partial + $FN_full_set_partial){
  $recall_full_set_partial =  sprintf("%.2f",($TP_full_set_partial / ($TP_full_set_partial + $FN_full_set_partial)) * 100);
} else {
  $recall_full_set_partial = sprintf("%.2f",0.00);
}

if ($precision_full_set_partial + $recall_full_set_partial){
  $f1_full_set_partial =   sprintf("%.2f",(2 * $precision_full_set_partial * $recall_full_set_partial) / ($precision_full_set_partial + $recall_full_set_partial));
} else {
  $f1_full_set_partial = sprintf("%.2f",0.00);
}


# print "coverage set content $span_set_coverage_content\n";
# print "total system content $total_content_system\n";
# print "total gold content $total_content_gold\n";

# print "coverage set source $span_set_coverage_source\n";
# print "total system source $total_source_system\n";
# print "total gold source $total_source_gold\n";

if ($total_content_system > 0){
  $precision_intersection_content = $span_set_coverage_content / $total_content_system;
} else {
  $precision_intersection_content = 0;
}

if($total_content_gold > 0){
  $recall_intersection_content = $span_set_coverage_content / $total_content_gold;
} else{
  $recall_intersection_content = 0;
}

if ($total_source_system > 0){
  $precision_intersection_source = $span_set_coverage_source / $total_source_system;
} else {
  $precision_intersection_source = 0;
}

if($total_source_gold > 0){
  $recall_intersection_source = $span_set_coverage_source / $total_source_gold;
} else{
  $recall_intersection_source = 0;
}


}## closes sub calculate results

#------------------------------------------------------
sub print_results{
#------------------------------------------------------


# print "intersection precision content $precision_intersection_content\n";
# print "intersection recall content $recall_intersection_content\n";


# print "intersection precision source $precision_intersection_source\n";
# print "intersection recall source $recall_intersection_source\n";

#print "----------------------+------+--------+------+------+------+---------------+------------+---------\n";
#print "-----------------------        S T R I C T          E V A L U A T I O N \n";
#print "----------------------+------+--------+------+------+------+---------------+------------+---------\n";
#print "                                         | gold | system | tp   | fp   | fn   | P (%) | R (%) | F1  (%) \n";
#print "----------------------+------+--------+------+------+------+---------------+------------+---------\n";
print "--------------------------------------------------------------------------------------------------------------------------------------\n"; 

printf ("%-45s  %8s   %8s   %8s   %8s   %8s   %8.2s   %8.2s   %8.2s\n", "", "gold", "system",  "TP", "FP", "FN",  "P",  "R", "F1");
print "--------------------------------------------------------------------------------------------------------------------------------------\n"; 


printf ("%-45s %8d | %8d | %8d | %8d | %8d | %8.2f | %8.2f | %8.2f\n", "Content full match", $total_content_gold , $total_content_system, $TP_content_only_strict, $FP_content_only_strict, $FN_content_only_strict,  $precision_content_strict, $recall_content_strict, $f1_content_strict);

printf ("%-45s %8d | %8d | %8d | %8d | %8d | %8.2f | %8.2f | %8.2f\n", "Content partial match", $total_content_gold , $total_content_system, $TP_content_only_partial, $FP_content_only_partial, $FN_content_only_partial,  $precision_content_partial, $recall_content_partial, $f1_content_partial);

printf ("%-45s %8d | %8d | %8s | %8s | %8s | %8.2f | %8.2f | %8.2s\n", "Content overlap-based", $total_content_gold , $total_content_system, "--",  "--",  "--",  $precision_intersection_content, $recall_intersection_content, "--");

print "--------------------------------------------------------------------------------------------------------------------------------------\n"; 

printf ("%-45s %8d | %8d | %8d | %8d | %8d | %8.2f | %8.2f | %8.2f\n", "Source full match",  $total_source_gold , $total_source_system, $TP_source_only_strict, $FP_source_only_strict, $FN_source_only_strict,  $precision_source_strict, $recall_source_strict, $f1_source_strict);

printf ("%-45s %8d | %8d | %8d | %8d | %8d | %8.2f | %8.2f | %8.2f\n", "Source partial match", $total_source_gold , $total_source_system, $TP_source_only_partial, $FP_source_only_partial, $FN_source_only_partial,  $precision_source_partial, $recall_source_partial, $f1_source_partial);

printf ("%-45s %8d | %8d | %8s | %8s | %8s | %8.2f | %8.2f | %8.2s\n", "Source overlap-based", $total_source_gold , $total_source_system, "--",  "--",  "--",  $precision_intersection_source, $recall_intersection_source, "--");
print "--------------------------------------------------------------------------------------------------------------------------------------\n"; 
#printf ("Cue full match: %23d | %6d | %4d | %4d | %4d | %13s | %10s | %7s\n",  $total_cue_gold , $total_cue_system, $TP_cue_only_strict, $FP_cue_only_strict, $FN_cue_only_strict,  $precision_cue_strict, $recall_cue_strict, $f1_cue_strict);

printf ("%-45s %8d | %8d | %8d | %8d | %8d | %8.2f | %8.2f | %8.2f\n", "Cue partial match",  $total_cue_gold , $total_cue_system, $TP_cue_only_partial, $FP_cue_only_partial, $FN_cue_only_partial,  $precision_cue_partial, $recall_cue_partial, $f1_cue_partial);

print "--------------------------------------------------------------------------------------------------------------------------------------\n";  
#printf ("Full set strict: %18d | %6d | %4d | %4d | %4d | %13s | %10s | %7s\n",  $total_full_set_gold , $total_full_set_system, $TP_full_set_strict, $FP_full_set_strict, $FN_full_set_strict,  $precision_full_set_strict, $recall_full_set_strict, $f1_full_set_strict);


printf ("%-45s %8d | %8d | %8d | %8d | %8d | %8.2f | %8.2f | %8.2f\n", "Content linked to source mention", $total_full_set_gold , $total_full_set_system, $TP_full_set_partial, $FP_full_set_partial, $FN_full_set_partial,  $precision_full_set_partial, $recall_full_set_partial, $f1_full_set_partial);
print "--------------------------------------------------------------------------------------------------------------------------------------\n"; 


}## closes sub print restuls


#-------------------------------------------
sub print_results_perfile{
#-------------------------------------------

print OUTFILE "--------------------------------------------------------------------------------------------------------------------------------------\n"; 

printf OUTFILE ("%-45s  %8s   %8s   %8s   %8s   %8s   %8.2s   %8.2s   %8.2s\n", "", "gold", "system",  "TP", "FP", "FN",  "P",  "R", "F1");
print OUTFILE "--------------------------------------------------------------------------------------------------------------------------------------\n"; 


printf OUTFILE ("%-45s %8d | %8d | %8d | %8d | %8d | %8.2f | %8.2f | %8.2f\n", "Content full match", $total_content_gold , $total_content_system, $TP_content_only_strict, $FP_content_only_strict, $FN_content_only_strict,  $precision_content_strict, $recall_content_strict, $f1_content_strict);

printf OUTFILE ("%-45s %8d | %8d | %8d | %8d | %8d | %8.2f | %8.2f | %8.2f\n", "Content partial match", $total_content_gold , $total_content_system, $TP_content_only_partial, $FP_content_only_partial, $FN_content_only_partial,  $precision_content_partial, $recall_content_partial, $f1_content_partial);

#printf OUTFILE ("%-45s %8d | %8d | %8s | %8s | %8s | %8.2f | %8.2f | %8.2s\n", "Content overlap-based", $total_content_gold , $total_content_system, "--",  "--",  "--",  $precision_intersection_content, $recall_intersection_content, "--");

print OUTFILE "--------------------------------------------------------------------------------------------------------------------------------------\n"; 

printf OUTFILE ("%-45s %8d | %8d | %8d | %8d | %8d | %8.2f | %8.2f | %8.2f\n", "Source full match",  $total_source_gold , $total_source_system, $TP_source_only_strict, $FP_source_only_strict, $FN_source_only_strict,  $precision_source_strict, $recall_source_strict, $f1_source_strict);

printf OUTFILE ("%-45s %8d | %8d | %8d | %8d | %8d | %8.2f | %8.2f | %8.2f\n", "Source partial match", $total_source_gold , $total_source_system, $TP_source_only_partial, $FP_source_only_partial, $FN_source_only_partial,  $precision_source_partial, $recall_source_partial, $f1_source_partial);

#printf OUTFILE ("%-45s %8d | %8d | %8s | %8s | %8s | %8.2f | %8.2f | %8.2s\n", "Source overlap-based", $total_source_gold , $total_source_system, "--",  "--",  "--",  $precision_intersection_source, $recall_intersection_source, "--");
print OUTFILE "--------------------------------------------------------------------------------------------------------------------------------------\n"; 
#printf OUTFILE ("Cue full match: %23d | %6d | %4d | %4d | %4d | %13s | %10s | %7s\n",  $total_cue_gold , $total_cue_system, $TP_cue_only_strict, $FP_cue_only_strict, $FN_cue_only_strict,  $precision_cue_strict, $recall_cue_strict, $f1_cue_strict);

printf OUTFILE ("%-45s %8d | %8d | %8d | %8d | %8d | %8.2f | %8.2f | %8.2f\n", "Cue partial match",  $total_cue_gold , $total_cue_system, $TP_cue_only_partial, $FP_cue_only_partial, $FN_cue_only_partial,  $precision_cue_partial, $recall_cue_partial, $f1_cue_partial);

print OUTFILE "--------------------------------------------------------------------------------------------------------------------------------------\n";  
#printf OUTFILE ("Full set strict: %18d | %6d | %4d | %4d | %4d | %13s | %10s | %7s\n",  $total_full_set_gold , $total_full_set_system, $TP_full_set_strict, $FP_full_set_strict, $FN_full_set_strict,  $precision_full_set_strict, $recall_full_set_strict, $f1_full_set_strict);


printf OUTFILE ("%-45s %8d | %8d | %8d | %8d | %8d | %8.2f | %8.2f | %8.2f\n", "Content linked to source mention", $total_full_set_gold , $total_full_set_system, $TP_full_set_partial, $FP_full_set_partial, $FN_full_set_partial,  $precision_full_set_partial, $recall_full_set_partial, $f1_full_set_partial);
print OUTFILE "--------------------------------------------------------------------------------------------------------------------------------------\n"; 


}## close print results per file


#------------------------------------------------------
#------------ compare arrays --------------------------
#------------------------------------------------------
#Paas it
#calc(\@array, $scalar)
#And then access it as: my @array = @{$_[0]};

sub compare_arrays {

my $found = 0;
my $compare = 1;

  @array1 =  @{$_[0]};
  @array2 =  @{$_[1]};

  for ($i=0;$i<=$#array1;$i++){
    $found = 0;
    for ($y=0;$y<=$#array2;$y++){
      if ($i==$y){
	$found = 1;
      }
    }
    if ($found == 0) {
      $i= $#array1 + 1;
      $compare = 0;
    }
  }

  for ($i=0;$i<=$#array2;$i++){
    $found = 0;
    for ($y=0;$y<=$#array1;$y++){
      if ($i==$y){
	$found = 1;
      }
    }
    if ($found == 0) {
      $i= $#array2 + 1;
      $compare = 0;
    }
  }

return $compare;

}






