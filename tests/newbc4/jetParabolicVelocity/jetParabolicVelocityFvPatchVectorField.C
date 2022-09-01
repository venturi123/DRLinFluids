/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2021 OpenFOAM Foundation
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

#include "jetParabolicVelocityFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::jetParabolicVelocityFvPatchVectorField::
jetParabolicVelocityFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchVectorField(p, iF),
    q_(0.0),
    omega_(0.0),
    r_(0.0),
    theta0_(0.0)
{
}


Foam::jetParabolicVelocityFvPatchVectorField::
jetParabolicVelocityFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchVectorField(p, iF),
    q_(dict.lookup<scalar>("Q")),
    omega_(dict.lookup<scalar>("omega")),
    r_(dict.lookupOrDefault<scalar>("r", 0.05)),
    theta0_(dict.lookup<scalar>("theta0"))
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


Foam::jetParabolicVelocityFvPatchVectorField::
jetParabolicVelocityFvPatchVectorField
(
    const jetParabolicVelocityFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchVectorField(ptf, p, iF, mapper),
    q_(ptf.q_),
    omega_(ptf.omega_),
    r_(ptf.r_),
    theta0_(ptf.theta0_)
{}


Foam::jetParabolicVelocityFvPatchVectorField::
jetParabolicVelocityFvPatchVectorField
(
    const jetParabolicVelocityFvPatchVectorField& ptf
)
:
    fixedValueFvPatchVectorField(ptf),
    q_(ptf.q_),
    omega_(ptf.omega_),
    r_(ptf.r_),
    theta0_(ptf.theta0_)
{}


Foam::jetParabolicVelocityFvPatchVectorField::
jetParabolicVelocityFvPatchVectorField
(
    const jetParabolicVelocityFvPatchVectorField& ptf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchVectorField(ptf, iF),
    q_(ptf.q_),
    omega_(ptf.omega_),
    r_(ptf.r_),
    theta0_(ptf.theta0_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::jetParabolicVelocityFvPatchVectorField::autoMap
(
    const fvPatchFieldMapper& m
)
{
    fixedValueFvPatchVectorField::autoMap(m);
}


void Foam::jetParabolicVelocityFvPatchVectorField::rmap
(
    const fvPatchVectorField& ptf,
    const labelList& addr
)
{
    fixedValueFvPatchVectorField::rmap(ptf, addr);

}


void Foam::jetParabolicVelocityFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    scalarField theta = atan(patch().Cf().component(vector::Y)() / patch().Cf().component(vector::X)());

    //scalarField a = q_ * constant::mathematical::pi / 2 / omega_ / sqr(r_) * cos(constant::mathematical::pi / omega_ * (theta - theta0_));

    scalarField a = q_ * cos(constant::mathematical::pi / omega_ * (theta - theta0_));

    fixedValueFvPatchVectorField::operator==
    (
        -a * (patch().Sf() / patch().magSf())
    );


    fixedValueFvPatchVectorField::updateCoeffs();

    //Info << "A = " << a << endl;
}


void Foam::jetParabolicVelocityFvPatchVectorField::write
(
    Ostream& os
) const
{
    fvPatchVectorField::write(os);
    writeEntry(os, "Q", q_);
    writeEntry(os, "omega", omega_);
    writeEntry(os, "r", r_);
    writeEntry(os, "theta0", theta0_);
    writeEntry(os, "value", *this);
}


// * * * * * * * * * * * * * * Build Macro Function  * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField
    (
        fvPatchVectorField,
        jetParabolicVelocityFvPatchVectorField
    );
}

// ************************************************************************* //
