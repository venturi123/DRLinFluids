/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "incrementallinearJetParabolicVelocityFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

Foam::scalar Foam::incrementallinearJetParabolicVelocityFvPatchVectorField::t() const
{
    return db().time().timeOutputValue();
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::incrementallinearJetParabolicVelocityFvPatchVectorField::
incrementallinearJetParabolicVelocityFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchVectorField(p, iF),
    omega_(0.0),
    r_(0.0),
    theta0_(0.0),
    q0_(0.0),
    q1_(0.0),
    alpha_(0.0),
    t0_(0.0),
    deltat_(0.0)
{
}


Foam::incrementallinearJetParabolicVelocityFvPatchVectorField::
incrementallinearJetParabolicVelocityFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchVectorField(p, iF),
    omega_(dict.lookup<scalar>("omega")),
    r_(dict.lookup<scalar>("r")),
    theta0_(dict.lookup<scalar>("theta0")),
    q0_(dict.lookup<scalar>("q0")),
    q1_(dict.lookup<scalar>("q1")),
    alpha_(dict.lookup<scalar>("alpha")),
    t0_(dict.lookup<scalar>("t0")),
    deltat_(dict.lookup<scalar>("deltaT"))
{


    fixedValueFvPatchVectorField::evaluate();

    /*
    // Initialise with the value entry if evaluation is not possible
    fvPatchVectorField::operator=
    (
        vectorField("value", dict, p.size())
    );
    */
}


Foam::incrementallinearJetParabolicVelocityFvPatchVectorField::
incrementallinearJetParabolicVelocityFvPatchVectorField
(
    const incrementallinearJetParabolicVelocityFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchVectorField(ptf, p, iF, mapper),
    omega_(ptf.omega_),
    r_(ptf.r_),
    theta0_(ptf.theta0_),
    q0_(ptf.q0_),
    q1_(ptf.q1_),
    alpha_(ptf.alpha_),
    t0_(ptf.t0_),
    deltat_(ptf.deltat_)
{}


Foam::incrementallinearJetParabolicVelocityFvPatchVectorField::
incrementallinearJetParabolicVelocityFvPatchVectorField
(
    const incrementallinearJetParabolicVelocityFvPatchVectorField& ptf
)
:
    fixedValueFvPatchVectorField(ptf),
    omega_(ptf.omega_),
    r_(ptf.r_),
    theta0_(ptf.theta0_),
    q0_(ptf.q0_),
    q1_(ptf.q1_),
    alpha_(ptf.alpha_),
    t0_(ptf.t0_),
    deltat_(ptf.deltat_)
{}


Foam::incrementallinearJetParabolicVelocityFvPatchVectorField::
incrementallinearJetParabolicVelocityFvPatchVectorField
(
    const incrementallinearJetParabolicVelocityFvPatchVectorField& ptf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchVectorField(ptf, iF),
    omega_(ptf.omega_),
    r_(ptf.r_),
    theta0_(ptf.theta0_),
    q0_(ptf.q0_),
    q1_(ptf.q1_),
    alpha_(ptf.alpha_),
    t0_(ptf.t0_),
    deltat_(ptf.deltat_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::incrementallinearJetParabolicVelocityFvPatchVectorField::autoMap
(
    const fvPatchFieldMapper& m
)
{
    fixedValueFvPatchVectorField::autoMap(m);
}


void Foam::incrementallinearJetParabolicVelocityFvPatchVectorField::rmap
(
    const fvPatchVectorField& ptf,
    const labelList& addr
)
{
    fixedValueFvPatchVectorField::rmap(ptf, addr);
}


void Foam::incrementallinearJetParabolicVelocityFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    scalarField theta = atan(patch().Cf().component(vector::Y)() / patch().Cf().component(vector::X)());

    //scalarField a = q_ * constant::mathematical::pi / 2 / omega_ / sqr(r_) * cos(constant::mathematical::pi / omega_ * (theta - theta0_));

    // scalar q_ = (q0_ - q1_) * pow(1 - alpha_, (t() - t0_) / deltat_ + 1) + q1_;
	scalar q_ = q0_ + (q1_ - q0_) / ((t() + deltat_ - t0_) / deltat_) ;

    scalarField a = q_ * cos(constant::mathematical::pi / omega_ * (theta - theta0_));

    fixedValueFvPatchVectorField::operator==
    (
        -a * (patch().Sf() / patch().magSf())
    );


    fixedValueFvPatchVectorField::updateCoeffs();
}


void Foam::incrementallinearJetParabolicVelocityFvPatchVectorField::write
(
    Ostream& os
) const
{
    fvPatchVectorField::write(os);
    writeEntry(os, "omega", omega_);
    writeEntry(os, "r", r_);
    writeEntry(os, "theta0", theta0_);
    writeEntry(os, "q0", q0_);
    writeEntry(os, "q1", q1_);
    writeEntry(os, "alpha", alpha_);
    writeEntry(os, "t0", t0_);
    writeEntry(os, "deltaT", deltat_);
    writeEntry(os, "value", *this);
{}
}


// * * * * * * * * * * * * * * Build Macro Function  * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField
    (
        fvPatchVectorField,
		incrementallinearJetParabolicVelocityFvPatchVectorField
    );
}

// ************************************************************************* //
